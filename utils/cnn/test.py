import glob
import torch
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

from train import BaseModel
from dataset import TestDataset

def parse_args():
    parser = ArgumentParser()
    
    # --optimizer vars
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--test_batch_size', type=int, default=32)
    # --dataset vars
    parser.add_argument('--image_dir', type=str, default='./open/0615_nms_images')
    parser.add_argument('--pth_dir', type=str, default='./trained_models/efficientnet_b0_230614_223933/best_wf1.pth')
    parser.add_argument('--num_classes', type=int, default=35)

    # --model vars
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')

    # --etc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saved_dir", type=str, default="./trained_models")

    args = parser.parse_args()

    args.device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'

    return args

def test(args, model, test_loader):
    model.eval()

    logits = []
    # logits = defaultdict(list)
    with torch.no_grad():
        for step, (inputs, file_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs = inputs.type(torch.float32).to(args.device)

            outputs = model(inputs)
            logits += outputs.detach().cpu().numpy().tolist()
            # for file_name, logit in zip(file_names, outputs):
            #     logits[str(file_name)] = logit.detach().cpu().numpy().tolist()

            
    return logits


if __name__=="__main__":
    args = parse_args()
    test_transform = A.Compose([
        A.Resize(
            height=256,
            width=256,
        ),
        A.Normalize(mean=(0.415, 0.433, 0.437), std=(0.182, 0.212, 0.233), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()
    ])

    image_dirs = glob.glob(args.image_dir + '/*.png')
    test_dataset = TestDataset(image_dirs=image_dirs, transforms=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1
    )
    model = BaseModel(
        num_classes=args.num_classes,
        model_name=args.model_name
    ).to(args.device)
    model.load_state_dict(torch.load(args.pth_dir))

    logits = test(args, model, test_loader)
    
    test_df = defaultdict(list)
    for image_dir, logit in zip(image_dirs, logits):
        image_name = image_dir.split('/')[-1]
        test_df['crop_name'].append(image_name)
        for i in range(35):
            test_df[i].append(logit[i])

    test_df = pd.DataFrame(test_df)
    test_df.to_csv(args.image_dir+'.csv', index=False)