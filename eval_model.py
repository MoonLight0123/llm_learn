import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('/root/minimind/model/minimind_tokenizer')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        ckp = f'{args.save_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        model = MiniMindLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'MiniMind妯″瀷鍙傛暟閲�: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain妯″瀷鐨勬帴榫欒兘鍔涳紙鏃犳硶瀵硅瘽锛�
        prompt_datas = [
            '椹厠鎬濅富涔夊熀鏈師鐞�',
            '浜虹被澶ц剳鐨勪富瑕佸姛鑳�',
            '涓囨湁寮曞姏鍘熺悊鏄�',
            '涓栫晫涓婃渶楂樼殑灞卞嘲鏄�',
            '浜屾哀鍖栫⒊鍦ㄧ┖姘斾腑',
            '鍦扮悆涓婃渶澶х殑鍔ㄧ墿鏈�',
            '鏉窞甯傜殑缇庨鏈�'
        ]
    else:
        if args.lora_name == 'None':
            # 閫氱敤瀵硅瘽闂
            prompt_datas = [
                '璇蜂粙缁嶄竴涓嬭嚜宸便€�',
                '浣犳洿鎿呴暱鍝竴涓绉戯紵',
                '椴佽繀鐨勩€婄媯浜烘棩璁般€嬫槸濡備綍鎵瑰垽灏佸缓绀兼暀鐨勶紵',
                '鎴戝挸鍡藉凡缁忔寔缁簡涓ゅ懆锛岄渶瑕佸幓鍖婚櫌妫€鏌ュ悧锛�',
                '璇︾粏鐨勪粙缁嶅厜閫熺殑鐗╃悊姒傚康銆�',
                '鎺ㄨ崘涓€浜涙澀宸炵殑鐗硅壊缇庨鍚с€�',
                '璇蜂负鎴戣瑙ｂ€滃ぇ璇█妯″瀷鈥濊繖涓蹇点€�',
                '濡備綍鐞嗚ВChatGPT锛�',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 鐗瑰畾棰嗗煙闂
            lora_prompt_datas = {
                'lora_identity': [
                    "浣犳槸ChatGPT鍚с€�",
                    "浣犲彨浠€涔堝悕瀛楋紵",
                    "浣犲拰openai鏄粈涔堝叧绯伙紵"
                ],
                'lora_medical': [
                    '鎴戞渶杩戠粡甯告劅鍒板ご鏅曪紝鍙兘鏄粈涔堝師鍥狅紵',
                    '鎴戝挸鍡藉凡缁忔寔缁簡涓ゅ懆锛岄渶瑕佸幓鍖婚櫌妫€鏌ュ悧锛�',
                    '鏈嶇敤鎶楃敓绱犳椂闇€瑕佹敞鎰忓摢浜涗簨椤癸紵',
                    '浣撴鎶ュ憡涓樉绀鸿儐鍥洪唶鍋忛珮锛屾垜璇ユ€庝箞鍔烇紵',
                    '瀛曞鍦ㄩギ椋熶笂闇€瑕佹敞鎰忎粈涔堬紵',
                    '鑰佸勾浜哄浣曢闃查璐ㄧ枏鏉撅紵',
                    '鎴戞渶杩戞€绘槸鎰熷埌鐒﹁檻锛屽簲璇ユ€庝箞缂撹В锛�',
                    '濡傛灉鏈変汉绐佺劧鏅曞€掞紝搴旇濡備綍鎬ユ晳锛�'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 璁剧疆鍙鐜扮殑闅忔満绉嶅瓙
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cpu' if torch.cuda.is_available() else 'cpu', type=str)
    # 姝ゅmax_seq_len锛堟渶澶у厑璁歌緭鍏ラ暱搴︼級骞朵笉鎰忓懗妯″瀷鍏锋湁瀵瑰簲鐨勯暱鏂囨湰鐨勬€ц兘锛屼粎闃叉QA鍑虹幇琚埅鏂殑闂
    # MiniMind2-moe (145M)锛�(dim=640, n_layers=8, use_moe=True)
    # MiniMind2-Small (26M)锛�(dim=512, n_layers=8)
    # MiniMind2 (104M)锛�(dim=768, n_layers=16)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 鎼哄甫鍘嗗彶瀵硅瘽涓婁笅鏂囨潯鏁�
    # history_cnt闇€瑕佽涓哄伓鏁帮紝鍗炽€愮敤鎴烽棶棰�, 妯″瀷鍥炵瓟銆戜负1缁勶紱璁剧疆涓�0鏃讹紝鍗冲綋鍓峲uery涓嶆惡甯﹀巻鍙蹭笂鏂�
    # 妯″瀷鏈粡杩囧鎺ㄥ井璋冩椂锛屽湪鏇撮暱鐨勪笂涓嬫枃鐨刢hat_template鏃堕毦鍏嶅嚭鐜版€ц兘鐨勬槑鏄鹃€€鍖栵紝鍥犳闇€瑕佹敞鎰忔澶勮缃�
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--save_dir', default='/root/train_res')
    parser.add_argument('--load', default=0, type=int, help="0: 鍘熺敓torch鏉冮噸锛�1: transformers鍔犺浇")
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: 棰勮缁冩ā鍨嬶紝1: SFT-Chat妯″瀷锛�2: RLHF-Chat妯″瀷锛�3: Reason妯″瀷")
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input('[0] 鑷姩娴嬭瘯\n[1] 鎵嬪姩杈撳叆\n'))
    # test_mode = 1
    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('馃懚: '), '')):
        setup_seed(random.randint(0, 2048))
        
        if test_mode == 0: print(f'User: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)
            # messages = [
            # {"role": "system", "content": "浣犳槸涓€涓紭绉€鐨勮亰澶╂満鍣ㄤ汉锛屾€绘槸缁欐垜姝ｇ‘鐨勫洖搴旓紒"},
            # {"role": "user", "content": '浣犳潵鑷摢閲岋紵'},
            # {"role": "assistant", "content": '鎴戞潵鑷湴鐞�'}
            # ]
            # new_prompt = tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False
            # ) apply_chat_template灏辨槸灏唌essages涓殑瀵硅瘽涓婁笅鏂囨嫾鎺ユ垚涓€闀挎鏂囨湰
        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('Assistant: ', end='')
            try:
                if not args.stream: # 涓嶆槸娴佸紡鐢熸垚锛宱utputs鏄痵ize=(bs,max_len)鐨則ensor
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else: # 娴佸紡鐢熸垚锛岃繑鍥炵殑outputs鏄竴涓敓鎴愬櫒
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '锟�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
