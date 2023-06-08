import glob
import cv2
import random
import os
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import cv2
import yaml
import torch
import shutil

from tqdm.auto import tqdm
from argparse import ArgumentParser

from pytz import timezone
from datetime import datetime

from ultralytics import YOLO

def parse_args():
    parser = ArgumentParser()
    
    # --optimizer vars
    parser.add_argument('--data', type=str, default='./data/yolo/custom.yaml')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model_dir', type=str, default='./yolov8x/230607_222740/weights/epoch240.pt')
    parser.add_argument('--project_name', type=str, default='yolov8x')

    parser.add_argument('--test_dir', type=str, default='../open/test/*.png')
    # parser.add_argument('--test_dir', type=str, default='../open/yolo/valid1/*.png')
    

    parser.add_argument('--imgsz_w', type=int, default=1920)
    parser.add_argument('--imgsz_h', type=int, default=1080)
    # parser.add_argument('--imgsz_w', type=int, default=1024)
    # parser.add_argument('--imgsz_h', type=int, default=555)
    parser.add_argument('--conf', type=float, default=0.6) # 0.25
    parser.add_argument('--iou', type=float, default=0.4) # 0.7
    parser.add_argument('--half', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--show', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--save_txt', type=bool, default=True)
    parser.add_argument('--save_conf', type=bool, default=True)
    parser.add_argument('--save_crop', type=bool, default=False)
    parser.add_argument('--max_det', type=int, default=100)
    parser.add_argument('--visualize', type=bool, default=True) # False
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--agnostic_nms', type=bool, default=True) # False
    parser.add_argument('--name', type=str, default='predict')
    parser.add_argument('--exist_ok', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=False)

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
def initialize():
    if os.path.exists("../open/yolo"):
        shutil.rmtree("../open/yolo")
        
    if not os.path.exists("../open/yolo/test"):
        os.makedirs("../open/yolo/test")    

def get_test_image_paths(test_image_paths, batch):
    for i in range(0, len(test_image_paths), batch):
        yield test_image_paths[i:i+batch]

def yolo_to_labelme(line, image_width, image_height, txt_file_name):
    file_name = txt_file_name.split("/")[-1].replace(".txt", ".png")
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]
    
    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)
    
    return file_name, int(class_id), confidence, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max



if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    
    # initialize()
    # make_yolo_dataset(sorted(glob("../open/test/*.png")), None, "test")

    model = YOLO(args.model_dir)
    test_image_paths = glob(args.test_dir)
    print(f'png 개수 = {len(test_image_paths)}')
    name = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d_%H%M%S")
    for i, images in tqdm(enumerate(get_test_image_paths(test_image_paths, args.batch)), total=int(len(test_image_paths)/args.batch)):
        model.predict(
            source=images,
            imgsz=(args.imgsz_w, args.imgsz_h),
            conf=args.conf,
            iou=args.iou,
            half=args.half,
            device=args.device,
            show=args.show,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            max_det=args.max_det,
            visualize=args.visualize,
            augment=args.augment,
            agnostic_nms=args.agnostic_nms,
            exist_ok=args.exist_ok, 
            verbose=args.verbose,
            
            project=args.project_name,
            name=name,
        )

    
    infer_txts = glob(f"{args.project_name}/{name}/labels/*.txt")
    print(f'txt 개수 = {len(infer_txts)}')

    # test_images = set([image_path.split('/')[-1].replace('.txt','') for image_path in test_image_paths])
    # test_annos = set([txt_path.split('/')[-1].replace('.png', '') for txt_path in infer_txts])
    # print(f'png - txt = {len(test_images - test_annos)}개: {test_images - test_annos}')

    results = []
    # [1920, 1080] must be multiple of max stride 32, updating to [1920, 1088]
    for infer_txt in tqdm(infer_txts):
        base_file_name = infer_txt.split("/")[-1].split(".")[0]
        # image_height, image_width = cv2.imread(f"../open/test/{base_file_name}.png").shape[:2]
        image_height, image_width = 1080, 1920
        with open(infer_txt, "r") as reader:        
            lines = reader.readlines()        
            for line in lines:
                results.append(yolo_to_labelme(line, image_width, image_height, infer_txt))

    df_submission = pd.DataFrame(data=results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
    df_submission.to_csv(f"{args.model_dir.replace('.pt','.csv')}", index=False)
    