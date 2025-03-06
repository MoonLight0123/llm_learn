最终得到的视觉模型的效果如下

<table>
  <thead>
    <tr>
      <th>图片</th>
      <th>Pretrain_vlm</th>
      <th>SFT_vlm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="test_img\eval_images\0.PNG" alt="sunset">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>在日落期间,美丽的云层和海景</td>
      <td>这幅图像的特点是一座大型海滩，位于一座高山上。这座大岛似乎是海岸线上一个明亮的天空，为它提供了一个美丽的背景。在画面中可以看到一座巨大的金色沙滩，为这个海滩增添了色彩和魅力。</td>
    </tr>
    <tr>
      <td>
        <img src="test_img\eval_images\1.PNG" alt="cloud">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>在蓝色的天空中,云层在天空中移动,时间流逝</td>
      <td>这幅图片描绘了一个美丽的天空，白云密布，天空中出现了一大片云。云朵散布在天空中，为画面增添了视觉趣味。天空中出现了不同颜色和大小的云朵，创造出一种美丽而宁静的气氛。</td>
    </tr>
    <tr>
      <td>
        <img src="test_img\eval_images\3.PNG" alt="bird">
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      </td>
      <td>一只鹰在飞行中,在空中飞翔</td>
      <td>这幅图像描绘了一个飞鸟和一只鸵鸟在天空中飞行。这只鸟很可能是在一片开阔的草地上飞行，它似乎正从树上飞过来，周围都是水和泥土。这只鸟很可能是在寻找食物或喂鸟，而这只鸟可能正从飞行中获得营养。</td>
    </tr>
  
  </tbody>
</table>

<!-- 本项目复现了 minimind-v 视觉语言模型的技术实现，成功构建了参数量仅 26M 的超轻量级多模态对话模型，该模型通过双阶段训练（视觉预训练 + 指令微调），在单卡 NVIDIA 3090 上仅需 1.3 元成本即可完成全流程训练（预训练 0.8h + 微调 0.2h），实现了视觉语言理解与对话的核心能力。 -->
本项目复现了 minimind-v 视觉语言模型的技术实现，成功构建了参数量仅 26M 的超轻量级多模态对话模型，该模型通过双阶段训练（视觉预训练 + 指令微调），实现了视觉语言理解与对话的核心能力。



[模型架构](model/model_vlm.py)上，采用[llm.md](llm.md)中的语言模型作为基座语言模型，仅增加了Visual Encoder与特征投影两个子模块，以支持多种模态信息输入。

具体来说，使用CLIP-ViT-Base作为Visual Encoder，特征投影方式采用与LlaVA-1相同的方式，即直接采用一个无偏的线性变换完成这个操作

<div style="text-align: center;">
    <img src="img/llava-structure.png" width="80%">
</div>

在单卡NVIDIA 4090完成[训练全过程](train_log)（Pretrain_vlm 2.5h + SFT_vlm 2h），模型训练后得到的所有权重文件均在

## 训练过程

采用类似llava两阶段训练的方式

### Pretain阶段
冻结...

### SFT阶段
冻结..
...