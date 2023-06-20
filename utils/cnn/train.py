import os
import math
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

import wandb
import numpy as np
from pytz import timezone
from datetime import datetime
from argparse import ArgumentParser
from importlib import import_module

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomDataset

import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = ArgumentParser()
    
    # --optimizer vars
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=9e-1)
    parser.add_argument('--optimizer', type=str, default='Adam')

    # --dataset vars
    parser.add_argument('--train_dataset', type=str, default='./open/5fold/train0.csv')
    parser.add_argument('--valid_dataset', type=str, default='./open/5fold/valid0.csv')
    # parser.add_argument('--valid_dataset', type=str, default='./fold1234_ft_nms_images.csv')
    parser.add_argument('--train34', type=str, default='./open/34_train.csv')
    parser.add_argument('--valid34', type=str, default='./open/34_valid.csv')
    # parser.add_argument('--num_classes', type=int, default=34)
    parser.add_argument('--num_classes', type=int, default=35)

    # --model vars
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')

    # --gradient_clipping
    parser.add_argument("--gradient_clipping", type=bool, default=False)

    # --etc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saved_dir", type=str, default="./trained_models")
    parser.add_argument("--amp", type=bool, default=True)

    # --wandb configs
    parser.add_argument("--wandb_project", type=str, default="dacon")
    parser.add_argument("--wandb_entity", type=str, default="wooyeolbaek")
    parser.add_argument("--wandb_run", type=str, default='exp')

    args = parser.parse_args()

    args.device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'

    args.wandb_run = args.model_name

    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class BaseModel(nn.Module):
    def __init__(self, num_classes, model_name, lib='torchvision.models'):
        super().__init__()

        if lib=='torchvision.models':
            self.backbone = nn.Sequential(
                getattr(import_module(lib), model_name)(weights="IMAGENET1K_V1"),
                nn.LazyLinear(num_classes)
            )
        # elif lib=='timm.create_model':
        #     self.backbone = getattr(import_module(lib))(model_name, pretrained=True, num_classes=num_classes)
        #     #self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)

        return x

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, criterion, optimizer, train_loader, valid_loader):
    seed_everything(args.seed)

    start_time = datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")
    saved_dir = os.path.join(args.saved_dir, args.wandb_run + start_time)

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # config 설정
    with open(os.path.join(saved_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # wandb 설정
    wandb.init(
        project=f"{args.wandb_project}",
        entity=f"{args.wandb_entity}",
        name=args.wandb_run + start_time,
    )
    wandb.config.update(
        {
            "run_name": args.wandb_run,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "valid_batch_size": args.valid_batch_size,
            "criterion": args.criterion,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "seed": args.seed,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
        }
    )

    model.to(args.device)
    criterion.to(args.device)

    # --AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_valid_f1 = 0

    with open(os.path.join(saved_dir,"log.txt"),"w") as f:
        for epoch in range(1,args.epochs+1):

            # --train
            model.train()
            train_loss = []
            num_train_batches = math.ceil(len(train_dataset) / args.train_batch_size)
            with tqdm(total=num_train_batches) as pbar:
                for step, (inputs, labels) in enumerate(train_loader):
                    pbar.set_description(f"[Train] Epoch [{epoch}/{args.epochs}]")

                    inputs = inputs.type(torch.float32).to(args.device)
                    labels = labels.to(args.device)

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    # gradient clipping
                    if args.gradient_clipping:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    mean_loss = loss.item() / len(labels)
                    train_loss.append(mean_loss)

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": round(mean_loss, 4),
                        }
                    )

            mean_train_loss = np.mean(train_loss)
            
            train_log = f"[EPOCH TRAIN {epoch}/{args.epochs}] : Train loss {mean_train_loss:4.2%}"
            f.write(train_log + "\n")

            results_dict = validation(args, epoch, model, criterion, valid_loader)
            f.write(results_dict["valid_log"] + "\n")
            
            if results_dict["f1_score"] >= best_valid_f1:
                print(f'New best model : {results_dict["f1_score"]:4.2%}! saving the best model..')
                torch.save(model.module.state_dict(), f"{saved_dir}/best_wf1.pth")
                best_valid_f1 = results_dict["f1_score"]
            
            torch.save(model.module.state_dict(), f"{saved_dir}/lastest.pth")
            print()

            wandb.log(
                {
                    "train/loss": round(mean_train_loss, 4),

                    "valid/loss": round(results_dict["mean_valid_loss"], 4),
                    "valid/f1_score": results_dict["f1_score"],
                    "valid/top1_score": results_dict["top1_score"],

                    # "iter": train_iteration,
                    "epoch": epoch,
                    "learning_rate": get_learning_rate(optimizer),
                }
            )

def validation(args, epoch, model, criterion, valid_loader):
    # val loop
    model.eval()
    valid_loss = []
    valid_top1 = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_classes, top_k=1).to(args.device)
    valid_f1 = torchmetrics.F1Score(task='multiclass', num_classes=args.num_classes, average='macro').to(args.device)

    with torch.no_grad():
        num_valid_batches = math.ceil(len(valid_dataset) / args.valid_batch_size)
        with tqdm(total=num_valid_batches) as pbar:
            for step, (inputs, labels) in enumerate(valid_loader):
                pbar.set_description(f"[Valid] Epoch [{epoch}/{args.epochs}]")

                inputs = inputs.type(torch.float32).to(args.device)
                labels = labels.to(args.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=-1)

                valid_top1.update(outputs, labels)
                valid_f1.update(preds, labels)
                mean_loss = loss.item() / len(labels)
                valid_loss.append(mean_loss)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": round(mean_loss, 4),
                        "acc": round(valid_top1.compute().item(), 4),
                        "f1": round(valid_f1.compute().item(), 4),
                    }
                )
                
        mean_valid_loss = np.mean(valid_loss)

        valid_top1_score = valid_top1.compute().item()
        valid_f1_score = valid_f1.compute().item()
        valid_top1.reset()
        valid_f1.reset()
        valid_log = f"[EPOCH VALID {epoch}/{args.epochs}] : Valid loss {mean_valid_loss:4.2%} \n Valid f1-score {valid_f1_score:4.2%} - Valid top1-score {valid_top1_score:4.2%}"
        results_dict = {
            "f1_score": valid_f1_score,
            "top1_score": valid_top1_score,
            "mean_valid_loss": mean_valid_loss,
            "valid_log": valid_log
        }
        return results_dict


if __name__ == '__main__':
    args = parse_args()

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.FancyPCA(
        #     alpha=0.1,
        #     always_apply=False,
        #     p=0.5
        # ),
        A.Resize(
            height=256,
            width=256,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(
            height=256,
            width=256,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])

    train_dataset = CustomDataset(csv_dir=args.train_dataset, new_class_dir=args.train34, transforms=train_transform)
    valid_dataset = CustomDataset(csv_dir=args.valid_dataset, new_class_dir=args.valid34, transforms=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=True,
        num_workers=1
    )

    model = BaseModel(
        num_classes=args.num_classes,
        model_name=args.model_name
    )
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    criterion = getattr(import_module("torch.nn"), args.criterion)()

    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        #momentum=args.momentum,
        #weight_decay=args.weight_decay,
    )
    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

    train(args, model, criterion, optimizer, train_loader, valid_loader)