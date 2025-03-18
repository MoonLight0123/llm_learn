import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
import deepspeed
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    # if dist.get_rank() == 0:
    print(content)
    with open('output.txt', 'a', encoding='utf-8') as file:
        print(content, file=file)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(engine.device)
        Y = Y.to(engine.device)
        loss_mask = loss_mask.to(engine.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        # with engine.training():
        res = engine(X)
        loss = loss_fct(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        loss += res.aux_loss
        engine.backward(loss)
        engine.step()

        # 学习率更新需要放在engine.step之后
        engine.optimizer.param_groups[0]['lr'] = lr

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    engine.optimizer.param_groups[0]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
                    
        if (step + 1) % args.save_interval == 0 and engine.local_rank == 0:
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}'
            engine.save_checkpoint(ckp)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('/kaggle/working/llm_learn/model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

def init_distributed_mode():
    deepspeed.init_distributed(dist_backend='nccl')  # 替换原来的dist初始化
    # global ddp_local_rank, DEVICE
    # ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # DEVICE = f"cuda:{ddp_local_rank}"
    # torch.cuda.set_device(DEVICE)
    torch.cuda.set_device(args.local_rank)
# def init_distributed_mode():
#     if not ddp: return
#     global ddp_local_rank, DEVICE

#     dist.init_process_group(backend="nccl")
#     ddp_rank = int(os.environ["RANK"])
#     ddp_local_rank = int(os.environ["LOCAL_RANK"])
#     ddp_world_size = int(os.environ["WORLD_SIZE"])
#     DEVICE = f"cuda:{ddp_local_rank}"
#     torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')

    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # parser.add_argument("--dtype", type=str, default="bfloat16")
    # parser.add_argument("--use_wandb", action="store_true")
    # parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    # parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=True, type=bool)
    parser.add_argument("--data_path", type=str, default="/kaggle/working/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # args.save_dir = os.path.join(args.out_dir)
    args.save_dir = './train_res'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    # ddp_local_rank, DEVICE = 0, "cuda:0"

    # if ddp:
    #     init_distributed_mode()
    #     args.device = torch.device(DEVICE)

    # if args.use_wandb and (not ddp or ddp_local_rank == 0):
    #     import wandb

    #     wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    # else:
    #     wandb = None
    
    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    engine, optimizer, train_loader, _ = deepspeed.initialize(
      model=model,
      model_parameters=parameters,
      training_data=train_ds,
      config=args.deepspeed_config,  # 从命令行参数获取配置文件路径
      dist_init_required=True
    )
    # train_sampler = DistributedSampler(train_ds) if ddp else None
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=args.batch_size,
    #     pin_memory=True,
    #     drop_last=False,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     sampler=train_sampler
    # )

    # scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # if ddp:
    #     model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
    #     model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)