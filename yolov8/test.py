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

from ultralytics import YOLO
from IPython.display import clear_output

def parse_args():
    parser = ArgumentParser()
    
    # --optimizer vars
    parser.add_argument('--data', type=str, default='./data/yolo/custom.yaml')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--model_dir', type=str, default='./object_detection_for_synthetic_data/exp/weights/best.pt')

    parser.add_argument('--imgsz_w', type=int, default=1920)
    parser.add_argument('--imgsz_h', type=int, default=1080)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7) # 0.2, 0.7
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

    # --wandb configs
    parser.add_argument("--wandb_project", type=str, default="object_detection_for_synthetic_data")
    parser.add_argument("--num_of_classes", type=int, default=34)
    parser.add_argument("--wandb_name", type=str, default="exp")

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

# 필요한 dir들 생성
def initialize():
    if os.path.exists("../open/yolo"):
        shutil.rmtree("../open/yolo")
        
    # if not os.path.exists("../open/yolo/valid"):
    #     os.makedirs("../open/yolo/valid")
        
    if not os.path.exists("../open/yolo/test"):
        os.makedirs("../open/yolo/test")    
        
    if not os.path.exists("./results"):
        os.makedirs("./results")

# 지정한 폴더의 png, txt 파일들로 yolo dataset 폴더를 만들어줌
def make_yolo_dataset(image_paths, txt_paths, type="train"):
    for image_path, txt_path in tqdm(zip(image_paths, txt_paths if not type == "test" else image_paths), total=len(image_paths)):
        source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)        
        image_height, image_width, _ = source_image.shape
        
        target_image_path = f"../open/yolo/{type}/{os.path.basename(image_path)}"
        cv2.imwrite(target_image_path, source_image)
        
        if type == "test":
            continue
        
        with open(txt_path, "r") as reader:
            yolo_labels = []
            for line in reader.readlines():
                line = list(map(float, line.strip().split(" ")))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x, y = float(((x_min + x_max) / 2) / image_width), float(((y_min + y_max) / 2) / image_height)
                w, h = abs(x_max - x_min) / image_width, abs(y_max - y_min) / image_height
                yolo_labels.append(f"{class_name} {x} {y} {w} {h}")
            
        target_label_txt = f"../open/yolo/{type}/{os.path.basename(txt_path)}"
        with open(target_label_txt, "w") as writer:
            for yolo_label in yolo_labels:
                writer.write(f"{yolo_label}\n")

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
    test_image_paths = glob("../open/yolo/test/*.png")
    for i, image in tqdm(enumerate(get_test_image_paths(test_image_paths, args.batch)), total=int(len(test_image_paths)/args.batch)):
        model.predict(
            source=image,
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
            project=args.wandb_project,
            name=args.name,
        )
        if i % 5 == 0:
            clear_output(wait=True)

    
    infer_txts = glob(f"{args.wandb_project}/predict/labels/*.txt")

    results = []
    for infer_txt in tqdm(infer_txts):
        base_file_name = infer_txt.split("/")[-1].split(".")[0]
        imgage_height, imgage_width = cv2.imread(f"../open/yolo/test/{base_file_name}.png").shape[:2]        
        with open(infer_txt, "r") as reader:        
            lines = reader.readlines()        
            for line in lines:
                results.append(yolo_to_labelme(line, imgage_width, imgage_height, infer_txt))

    df_submission = pd.DataFrame(data=results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
    df_submission.to_csv(f"./results/{args.wandb_project}.csv", index=False)