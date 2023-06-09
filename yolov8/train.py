import json
import cv2
import random
from glob import glob
import os
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings(action='ignore')

import yaml
import shutil
import torch
import numpy as np
from tqdm.auto import tqdm
from argparse import ArgumentParser

from pytz import timezone
from datetime import datetime

from pycocotools.coco import COCO
from ultralytics import YOLO

def parse_args():
    parser = ArgumentParser()
    # datasets
    parser.add_argument("--train_json_dir", type=str, default="../open/5fold/train3.json")
    parser.add_argument("--valid_json_dir", type=str, default="../open/5fold/valid3.json")
    parser.add_argument('--dataset_yml_dir', type=str, default='../open/yolo/train_yaml.yaml')

    parser.add_argument("--model", type=str, default="yolov8x")
    
    # --model keys
    parser.add_argument('--imgsz_w', type=int, default=1024)
    parser.add_argument('--imgsz_h', type=int, default=555)
    parser.add_argument('--epochs', type=int, default=260)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_period', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    #parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exist_ok', type=bool, default=True)
    #parser.add_argument('--project', type=str, default='yolo')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr0', type=float, default=3e-3)
    # parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--cache', type=bool, default=True)
    parser.add_argument('--cos_lr', type=bool, default=True)
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('--lrf', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_momentum', type=float, default=0.8)
    parser.add_argument('--warmup_bias_lr', type=float, default=3e-4)

    parser.add_argument('--box', type=float, default=7.5) # default 7.5
    parser.add_argument('--cls', type=float, default=0.5) # default 0.5
    parser.add_argument('--dfl', type=float, default=1.5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--nbs', type=int, default=16)


    parser.add_argument("--save_json", type=bool, default=True)
    parser.add_argument("--save_hybrid", type=bool, default=True)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--max_det', type=float, default=10)
    parser.add_argument("--half", type=bool, default=True)
    parser.add_argument("--plots", type=bool, default=True)
    parser.add_argument("--rect", type=bool, default=False)

    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.1)
    parser.add_argument("--translate", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--shear", type=float, default=0.1)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.3)
    parser.add_argument("--mosaic", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--copy_paste", type=float, default=0.5)

    args = parser.parse_args()


    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


classes = [
    'chevrolet_malibu_sedan_2012_2016',
    'chevrolet_malibu_sedan_2017_2019',
    'chevrolet_spark_hatchback_2016_2021',
    'chevrolet_trailblazer_suv_2021_',
    'chevrolet_trax_suv_2017_2019',
    'genesis_g80_sedan_2016_2020',
    'genesis_g80_sedan_2021_',
    'genesis_gv80_suv_2020_',
    'hyundai_avante_sedan_2011_2015',
    'hyundai_avante_sedan_2020_',
    'hyundai_grandeur_sedan_2011_2016',
    'hyundai_grandstarex_van_2018_2020',
    'hyundai_ioniq_hatchback_2016_2019',
    'hyundai_sonata_sedan_2004_2009',
    'hyundai_sonata_sedan_2010_2014',
    'hyundai_sonata_sedan_2019_2020',
    'kia_carnival_van_2015_2020',
    'kia_carnival_van_2021_',
    'kia_k5_sedan_2010_2015',
    'kia_k5_sedan_2020_',
    'kia_k7_sedan_2016_2020',
    'kia_mohave_suv_2020_',
    'kia_morning_hatchback_2004_2010',
    'kia_morning_hatchback_2011_2016',
    'kia_ray_hatchback_2012_2017',
    'kia_sorrento_suv_2015_2019',
    'kia_sorrento_suv_2020_',
    'kia_soul_suv_2014_2018',
    'kia_sportage_suv_2016_2020',
    'kia_stonic_suv_2017_2019',
    'renault_sm3_sedan_2015_2018',
    'renault_xm3_suv_2020_',
    'ssangyong_korando_suv_2019_2020',
    'ssangyong_tivoli_suv_2016_2020'
]

# 필요한 dir들 생성
def initialize(args):
    if not os.path.exists("../open/yolo"):
        os.makedirs("../open/yolo")

    if not os.path.exists(f"../open/yolo/{args.train_json_dir.split('/')[-1].replace('.json', '')}"):
        os.makedirs(f"../open/yolo/{args.train_json_dir.split('/')[-1].replace('.json', '')}")
        
    if not os.path.exists(f"../open/yolo/{args.valid_json_dir.split('/')[-1].replace('.json', '')}"):
        os.makedirs(f"../open/yolo/{args.valid_json_dir.split('/')[-1].replace('.json', '')}")
        


def coco2yolo(json_dir, image_prefix='train', base_dir='../open', test=False):
    # 저장할 폴더 생성
    json_name = json_dir.split('/')[-1].replace('.json', '')
    yolo_base_dir = os.path.join(base_dir, 'yolo', json_name)
    if not os.path.exists(yolo_base_dir):
        os.makedirs(yolo_base_dir)

    # coco json 파일 -> yolo txt 파일로 변환
    coco = COCO(json_dir)
    for image_id in tqdm(coco.getImgIds(), total=len(coco.getImgIds())):
        image_info = coco.loadImgs(image_id)[0]
        image_name = image_info['file_name']

        # 이미지 이동
        image_dir = os.path.join(base_dir, image_prefix, image_name)
        yolo_image_dir = os.path.join(yolo_base_dir, image_name)
        image = cv2.imread(image_dir)
        cv2.imwrite(yolo_image_dir, image)

        # annotation이 없는 경우
        if test:
            continue

        # yolo label 생성
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        image_w = image_info['width']
        image_h = image_info['height']
        yolo_labels = []
        for ann in anns:
            category_id = ann['category_id']
            xmin, ymin, width, height = ann['bbox']
            x = (2*xmin + width) / (2 * image_w)
            y = (2*ymin + height) / (2 * image_h)
            w = width / image_w
            h = height / image_h

            yolo_labels.append(f'{category_id} {x:.5f} {y:.5f} {w:.5f} {h:.5f}')

        # yolo label 저장
        yolo_dir = os.path.join(yolo_base_dir, image_name.replace('.png', '.txt'))
        with open(yolo_dir, 'w') as f:
            f.write('\n'.join(yolo_labels))


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    # initialize
    initialize(args)
    if not os.path.exists(args.dataset_yml_dir):
        coco2yolo(json_dir=args.train_json_dir, image_prefix='train', base_dir='../open')
        coco2yolo(json_dir=args.valid_json_dir, image_prefix='train', base_dir='../open')
        train_yaml = {
            "names": classes,
            "nc": len(classes),
            # "path": "/Users/wooyeolbaek/Downloads/untitled_folder/object-detection-for-synth-data/open/yolo",
            "path": "/opt/ml/object-detection-for-synthetic-data/open/yolo",
            # "path": "/home/elicer/open/yolo",
            "train": f'{args.train_json_dir.split("/")[-1].replace(".json", "")}',
            "val": f'{args.valid_json_dir.split("/")[-1].replace(".json", "")}',
        }

        with open(args.dataset_yml_dir, "w") as writer:
            yaml.dump(train_yaml, writer)
    
    #model = YOLO(f"{MODEL}/train/weights/last.pt")
    model = YOLO(args.model)
    results = model.train(
        data=args.dataset_yml_dir,
        imgsz=(args.imgsz_w, args.imgsz_h),
        save=True,
        save_period=args.save_period,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=args.exist_ok,
        name=datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d_%H%M%S"),
        #name=args.wandb_name,
        project=args.model,
        seed=args.seed,
        pretrained=args.pretrained,
        resume=args.resume,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        amp=args.amp,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        label_smoothing=args.label_smoothing,
        nbs=args.nbs,
        
        #augment=args.augment,
        # save_json=args.save_json,
        # save_hybrid=args.save_hybrid,
        # conf=args.conf,
        # iou=args.iou,
        # max_det=args.max_det,
        # half=args.half,
        # plots=args.plots,
        # rect=args.rect,
        
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
    )