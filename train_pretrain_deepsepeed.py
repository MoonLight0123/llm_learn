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
    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        print(content)
        with open('output.txt', 'a', encoding='utf-8') as file:
            print(content, file=file)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, engine, train_loader, args, lm_config):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    iter_per_epoch = len(train_loader)
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(engine.device)
        Y = Y.to(engine.device)
        loss_mask = loss_mask.to(engine.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        engine.train()
        res = engine(X)
        loss = loss_fct(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        loss += res.aux_loss
        
        engine.backward(loss)
        engine.step()

        # 学习率更新
        engine.optimizer.param_groups[0]['lr'] = lr

        if step % args.log_interval == 0 and engine.local_rank == 0:
            spend_time = time.time() - start_time
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) '
                f'loss:{loss.item():.3f} lr:{lr:.12f} '
                f'epoch_Time:{spend_time/(step+1)*iter_per_epoch//60 - spend_time//60}min'
            )
                    
        if (step + 1) % args.save_interval == 0 and engine.local_rank == 0:
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}'
            engine.save_checkpoint(ckp)

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('/kaggle/working/llm_learn/model/minimind_tokenizer')
    model = MiniMindLM(lm_config)
    if dist.get_rank() == 0:
        Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument('--deepspeed_config', type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)  # DeepSpeed自动处理
    args = parser.parse_args()

    # 初始化配置
    lm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    
    args.save_dir = './train_res'
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(1337)

    # 初始化模型和数据集
    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)

    # DeepSpeed初始化
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        dist_init_required=True,
        training_data=train_ds  # 让DeepSpeed自动处理数据加载
    )

    # 获取DeepSpeed自动创建的DataLoader
    train_loader = engine.training_dataloader

    # 训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, engine, train_loader, args, lm_config)