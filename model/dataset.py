import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast
from PIL import Image
from model.model_vlm import MiniMindVLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VLMDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):
        # vit-base-patch16视觉编码器，因此用16*16=196个imagetoken
        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
                # 数据集格式如下：
                # {"conversations": 
                #  [
                #   {"role": "user", "content": "提供给定图像的简要描述。\n<image>"}, 
                #   {"role": "assistant", "content": "橄榄油是自由使用的健康成分。"}
                #  ], 
                #  "image": "GCC_train_002582585.jpg"
                # }
                
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        # 下面是构造的对话模板
        # <s>system
        # 你是 MiniMind，是一个有用的人工智能助手。</s>
        # <s>user
        # 提供给定图像的简要描述。
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@</s>
        # <s>assistant
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return X, Y, loss_mask, image_tensors


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)\
        # print(samples[:3])
        # [{'text': '<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s> <s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s> <s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s> <s>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。</s> <s>帮我想一个有趣的标题。这个挺有趣的："如何成为一名成功的魔术师" 调皮的标题往往会吸引读者的注意力。</s> <s>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。</s> <s>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。</s>'}, 
        # {'text': '<s>根据输入的内容，编写一个类别标签。\n这是一篇介绍如何阅读心电图的文章类别标签: 医学/心电图阅读指南</s> <s>帮我搜索一下最近的天气情况。当然，我可以帮您搜索最新的天气情况。请问您需要查询哪个城市的天气情况呢？</s> <s>帮我讲一个令人开心的笑话。好的，我帮您讲一个关于细菌的笑话。为什么细菌不会上网？因为连接总是断开了！</s> <s>现在给我生成一首关于大海的五言诗。碧波万顷月满天，海天相接处天地间。波涛滚滚江山美，海鸟翱翔日月闲。</s> <s>谢谢你，这篇文章很有用。不客气，我很高兴能够为您提供帮助。如果您还有其他问题或需求，随时可以对我说。</s> <s>你好，我想下载一个视频编辑软件，你有什么推荐吗？您好！当然，有很多选择。您想要免费软件还是愿意付费？</s> <s>为什么我的程序不输出正确结果？可能是代码逻辑有误，或者输入数据有误，需要仔细调试代码逻辑和输入数据。</s> <s>谢谢你的回答。现在我想知道这场比赛的具体时间和地点。这场比赛的时间是北京时间10月4日，地点是上海。</s>'}, 
        # {'text': '<s>客户要求看一份报告，但他没有说明他需要哪份报告。他是指哪份报告？这个问题没有指代消解，需进一步询问。</s> <s>鉴别并描述一处房地产广告中的行骗手段。作为语言模型AI，我不支持或鼓励行骗行为，故无法为此作出回答。</s> <s>表现一下植物界中的“四大名捕”分别是哪些。抱茎葛、卷丹、吊兰和狗尾草可以被称为植物界的“四大名捕”。</s> <s>回答以下关于特定主题的开放式问题。好的，请告诉我您想了解哪个主题，我会尽力以详细的方式回答您的问题。</s> <s>你好，我需要帮助解决一个问题。您好，欢迎来到我们的平台。请告诉我您遇到的问题，我将尽力为您提供帮助。</s> <s>为什么我的程序速度很慢？可能是由于算法复杂度不佳导致程序执行速度较慢，请仔细考虑算法改进优化的问题。</s> <s>请编写一段人机对话程序，模拟用户和AI机器人的聊天过程。AI机器人：您好，请问有什么我可以帮忙的吗？</s> <s>能否将该句子进一步近义词替换为另一种表达方式呢？当他表现得如此出色时，我们得到了非常令人惊喜的结果。</s>'}]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # print(encoding)
        # print(encoding.input_ids.shape)
        # torch.Size([1, 512]) 因此squeeze
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                # print(data)
                # {'conversations': 
                #  [{'role': 'user', 'content': '请告诉我在中国古代的“四大发明”是什么？'},
                #  {'role': 'assistant', 'content': '中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：\n\n1. **造纸术**：据史书记载，东汉时期的蔡伦改进了造纸工艺，使得纸张的生产更加便捷、成本更低，质量也更加优良。这一发明极大地促进了文化的传播和保存。\n\n2. **印刷术**：中国古代的印刷术最早可以追溯到唐代的雕版印刷，到了宋代发展出了活字印刷技术。印刷术的发明极大地促进了知识的传播，降低了书籍的成本，对教育和文化的发展起到了重要的推动作用。\n\n3. **火药**：火药最初是在唐代被发现的，最初可能用于医疗或炼金术。到了宋代，人们开始将火药用于军事目的，发明了各种火器。火药的发明改变了战争的面貌，对世界军事技术的发展产生了重大影响。\n\n4. **指南针**：指南针最初在中国被用于风水测量，后来逐渐发展成为航海导航的重要工具。这一发明极大地促进了海上航行技术的发展，对于新航路的开辟和世界地理大发现起到了关键作用。\n\n这四项发明不仅在中国历史上占有重要地位，而且对全世界的科技进步和文明发展都产生了深远的影响。'}]
                # }
                # break
                # 下面是reason distill时的数据格式 由于数据中加入了<think>与<answer>标签，所以回答时也会按照相应的格式回答
                # {"conversations": 
                #     [
                #      {"role": "user", "content": "请用一段话描述阿里巴巴集团的企业文化。"}, 
                #      {"role": "assistant", "content": 
                #          "<think>\n嗯，用户让我用一段话描述阿里巴巴集团的企业文化。首先，我需要明确阿里巴巴集团的平均文化是什么。企业文化的定义是企业内部发生、指导管理方向的准则和规范性的精神或规范。\n\n阿里巴巴集团在2000年代中期开始发展，之前也有类似的概念，比如微软的人工智能理念。所以，我可以从这个发展中来寻找关键词和结构。\n\n接下来，我要考虑用户可能的意图。用户可能想了解阿里巴巴的企业文化，或者需要用于某个报告、宣传材料或者内部交流中。因此，我需要确保描述既全面又有深度，涵盖战略、创新、效率等各个方面。\n\n我还需要用简洁的语言来表达，避免过于复杂的词汇，让段落易于理解。同时，要注意逻辑连贯，段落之间要有过渡，让读者能够顺畅地跟随观点。\n\n可能还需要引用一些关键点，比如战略协作、用户体验、创新、效率、数字化转型、客户支持、全球化等，这些都是阿里巴巴成功的关键点，能够帮助用户全面了解企业文化。\n\n最后，检查语法和结构是否正确，确保专业性和准确性，这样用户可以直接使用这段文字，或者稍作修改，就能得到满意的效果。\n</think>\n
                #          <answer>\n阿里巴巴集团的企业文化以战略协作为核心，倡导员工之间的资源整合与创新思维的碰撞，旨在实现企业价值的提升和持续发展。阿里巴巴将其视为一个协同共生的整体，致力于通过技术创新、用户体验优化和数字化转型，致力于为客户创造最大价值。企业内外部力量协同作战，推动企业从单纯的业务拓展延伸至价值创造和社会服务，并在全球范围内进行全方位的数字化转型，以满足多样化、个性化、高端化的客户需求。阿里巴巴集团ix platform的建立旨在帮助员工实现高效协作，激发创新精神，吸引更多优秀人才加入，共同推动企业不断向前发展。\n</answer>"
                #      }
                #     ]
                # }
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # print(prompt)
        # <s>system
        # 你是 MiniMind，是一个有用的人工智能助手。</s>
        # <s>user
        # 请用一段话描述阿里巴巴集团的企业文化。</s>
        # <s>assistant
        # 阿里巴巴集团的企业文化以“客户第一、员工第二、股东第三”为核心价值观，强调“让天下没有难做的生意”的使命。公司倡导开放、透明、分享、责任的团队合作精神，鼓励员工创新、追求卓越，同时注重员工的个人成长和幸福感。阿里巴巴的企业文化还体现在其独特的“六脉神剑”价值观体系中，包括客户第一、拥抱变化、团队合作、诚信、激情、专业等六个方面，这些价值观不仅指导着公司的日常运营，也深深影响着每一位阿里人的行为准则。</s>
        # 这里的第一句system消息是由设置好的tokenlizer的配置文件中设置好的
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)
                # print(type(obj))
                # break
        # dpo数据集中一行数据如下
        # {"chosen":
        #   [
        #   {"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", 
        #   "role": "user"}, 
        #   {"content": "To determine the number of moles of HBr required to react with 2 moles of C2H6 (ethane) to form 2 moles of C2H5Br (ethyl bromide) and 2 moles of H2, we need to write the balanced chemical equation for the reaction.\n\nThe reaction is a halogenation reaction, specifically the reaction of ethane with hydrogen bromide (HBr) to form ethyl bromide and hydrogen gas. The balanced chemical equation for this reaction is:\n\n\\[ \\text{C}_2\\text{H}_6 + \\text{HBr} \\rightarrow \\text{C}_2\\text{H}_5\\text{Br} + \\text{H}_2 \\]\n\nHowever, this equation is not balanced as written. To balance it, we need to ensure that the number of atoms of each element on the reactant side is equal to the number of atoms of that element on the product side. The balanced equation is:\n\n\\[ \\text{C}_2\\text{H}_6 + \\text{HBr} \\rightarrow \\text{C}_2\\text{H}_5\\text{Br} + \\text{H}_2 \\]\n\nUpon closer inspection, we see that the equation is already balanced as written, with 1 mole of C2H6 reacting with 1 mole of HBr to produce 1 mole of C2H5Br and 1 mole of H2.\n\nGiven that you have 2 moles of C2H6, and the stoichiometry of the reaction is 1:1 for C2H6 to HBr, you would need 2 moles of HBr to react with 2 moles of C2H6 to form 2 moles of C2H5Br and 2 moles of H2.\n\nTherefore, the number of moles of HBr required is **2 moles**.", 
        #   "role": "assistant"}
        #   ], 
        
        # "rejected": 
        #   [
        #     {"content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?", 
        #     "role": "user"}, 
        #     {"content": "To answer this question, we need to write down the chemical equation representing the reaction between hydrogen bromide (HBr) and ethane (C2H6):\n\nC2H6(g) + 2HBr(g) \u2192 C2H5Br(l) + H2(g)\n\nNow, we can set up a mole ratio based on the coefficients in front of each substance in the balanced equation:\n\nMoles of HBr : Moles of C2H6 :: 2 : 1\n\nWe also know that there are two moles of C2H6 present in our starting mixture, so we can substitute this value into the above mole ratio to determine the amount of HBr needed:\n\n(2 mol C2H6)/(1 mol C2H6) x 2 mol HBr = 4 mol HBr\n\nSo, four moles of HBr would be required to completely react with two moles of C2H6 according to the given stoichiometric relationship.", 
        #     "role": "assistant"}
        # ]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        # print(chosen_prompt)
        # <s>system
        # 你是 MiniMind，是一个有用的人工智能助手。</s>
        # <s>user
        # How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?</s>
        # <s>assistant
        # To determine the number of moles of HBr required to react with 2 moles of C2H6 (ethane) to form 2 moles of C2H5Br (ethyl bromide) and 2 moles of H2, we need to write the balanced chemical equation for the reaction.

        # The reaction is a halogenation reaction, specifically the reaction of ethane with hydrogen bromide (HBr) to form ethyl bromide and hydrogen gas. The balanced chemical equation for this reaction is:

        # \[ \text{C}_2\text{H}_6 + \text{HBr} \rightarrow \text{C}_2\text{H}_5\text{Br} + \text{H}_2 \]

        # However, this equation is not balanced as written. To balance it, we need to ensure that the number of atoms of each element on the reactant side is equal to the number of atoms of that element on the product side. The balanced equation is:

        # \[ \text{C}_2\text{H}_6 + \text{HBr} \rightarrow \text{C}_2\text{H}_5\text{Br} + \text{H}_2 \]

        # Upon closer inspection, we see that the equation is already balanced as written, with 1 mole of C2H6 reacting with 1 mole of HBr to produce 1 mole of C2H5Br and 1 mole of H2.

        # Given that you have 2 moles of C2H6, and the stoichiometry of the reaction is 1:1 for C2H6 to HBr, you would need 2 moles of HBr to react with 2 moles of C2H6 to form 2 moles of C2H5Br and 2 moles of H2.

        # Therefore, the number of moles of HBr required is **2 moles**.</s>
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        # print(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

if __name__ == "__main__":
    pass
