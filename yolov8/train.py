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

from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# import wandb
# from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="yolov8x")
    
    # --model keys
    parser.add_argument('--data', type=str, default='./data/yolo/custom.yaml')
    parser.add_argument('--imgsz_w', type=int, default=512)
    parser.add_argument('--imgsz_h', type=int, default=277)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    #parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exist_ok', type=bool, default=True)
    #parser.add_argument('--project', type=str, default='yolo')
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr0', type=float, default=1e-3)
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--cache', type=bool, default=True)
    parser.add_argument('--cos_lr', type=bool, default=True)
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('--lrf', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--warmup_momentum', type=float, default=0.8)
    parser.add_argument('--warmup_bias_lr', type=float, default=1e-4)
    parser.add_argument("--project_name", type=str, default="v2")

    parser.add_argument("--save_json", type=bool, default=True)
    parser.add_argument("--save_hybrid", type=bool, default=True)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--max_det', type=float, default=50)
    parser.add_argument("--half", type=bool, default=True)
    parser.add_argument("--plots", type=bool, default=True)
    parser.add_argument("--rect", type=bool, default=False)

    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--copy_paste", type=float, default=0.2)

    # --wandb configs
    parser.add_argument("--wandb_project", type=str, default="object_detection_for_synthetic_data")
    parser.add_argument("--num_of_classes", type=int, default=34)
    parser.add_argument("--wandb_name", type=str, default="exp")

    args = parser.parse_args()


    return args

def wandb_init(args):
    config = {
        "project": args.wandb_project,
        "num_of_classes": args.num_of_classes,
    }

    run = wandb.init(
        project=args.wandb_project,
        config=config,
        name=args.wandb_name,
    )

    artifact = wandb.Artifact(
        name = "train_data",
        type = "dataset",
    )

    artifact.add_dir("../open/yolo/train")
    wandb.log_artifact(artifact)

def get_class_labels():
    with open("../open/classes.txt", "r") as reader:
        lines = reader.readlines()
        class_labels = {int(line.strip().split(",")[0]):line.strip().split(",")[1] for line in lines}

    return class_labels


def wandb_run(args):
    config = {
        "project": args.wandb_project,
        "num_of_classes": args.num_of_classes,
    }

    run = wandb.init(
        project=args.wandb_project,
        config=config,
        name=args.wandb_name,
    )

    return run

def box_dict_maker(no_of_times, bounding_box, class_labels):
    box_list = []
    class_num = 0
    for n in range(no_of_times):
        intermediate = {
                    "position": {
                    "middle" : [float(bounding_box[5*n + 1]),float(bounding_box[5*n + 2])],
                    "width" : float(bounding_box[5*n + 3]),
                    "height" : float(bounding_box[5*n +4]),
                    },
                    "class_id" : int(bounding_box[5*n + 0]),
                    "box_caption": class_labels[int(bounding_box[5*n + 0])],
        }
        class_num = int(bounding_box[5*n + 0])
        box_list.append(intermediate)
    return (class_num, box_list)

def bounding_box_fn(file_name, class_labels):
    box = []
    box_dict = {}
    with open(file_name) as f:
        for w in f.readlines():
            for l in w.split(" "):
                box.append(l)
    no_of_times = int(len(box) / 5)
    class_num, box_list = box_dict_maker(no_of_times, box, class_labels)
    final_dict = {
        "ground_truth" : {
            "box_data" : box_list,
            "class_labels" : class_labels
        }
    }
    return (class_num, final_dict)

def execute(PATH_IMAGE, PATH_TEXT, NAME, class_labels, run):
    NAME_LIST = []
    for x in os.listdir(PATH_TEXT):
        if x.endswith(".txt"):
            NAME_LIST.append(x[:-4])
    tabular_data = []
    count = 0
    for x in NAME_LIST:
        box_path = PATH_TEXT + str(x) + ".txt"
        image_path = PATH_IMAGE + str(x) + ".png"
        class_num, final_dict = bounding_box_fn(box_path, class_labels)
        tabular_data.append([count, wandb.Image(image_path,
        boxes = final_dict), class_labels[class_num]])
        count += 1
    columns = ['index', 'image', 'label']
    test_table = wandb.Table(data = tabular_data, columns = columns)
    run.log({NAME : test_table})

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

    if not os.path.exists("../open/yolo/train"):
        os.makedirs("../open/yolo/train")
        
    if not os.path.exists("../open/yolo/valid"):
        os.makedirs("../open/yolo/valid")
        
    # if not os.path.exists("../open/yolo/test"):
    #     os.makedirs("../open/yolo/test")
        
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

    # # initialize
#     initialize()
#     image_paths = sorted(glob("../open/train/*.png"))
#     txt_paths = sorted(glob("../open/train/*.txt"))

#     train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths = train_test_split(image_paths, txt_paths, test_size=0.1, random_state=args.seed)

#     make_yolo_dataset(train_images_paths, train_txt_paths, "train")
#     make_yolo_dataset(valid_images_paths, valid_txt_paths, "valid")

#     with open("../open/classes.txt", "r") as reader:
#         lines = reader.readlines()
#         classes = [line.strip().split(",")[1] for line in lines]

#     yaml_data = {
#                 "names": classes,
#                 "nc": len(classes),
#                 # "path": "/Users/wooyeolbaek/Downloads/untitled_folder/object-detection-for-synth-data/open/yolo",
#                 # "path": "/opt/ml/object-detection-for-synthetic-data/open/yolo",
#                 "path": "/home/elicer/open/yolo",
#                 "train": "train",
#                 "val": "valid",
#                 "test": "test"
#                 }

#     with open("../open/yolo/custom.yaml", "w") as writer:
#         yaml.dump(yaml_data, writer)

    # # wandb init
    # wandb_init(args)
    # class_labels = get_class_labels()
    # run = wandb_run(args)
    # PATH_TRAIN_IMAGES = "../open/yolo/train/"
    # PATH_TRAIN_LABELS = "../open/yolo/train/"
    # PATH_VAL_IMAGES = "../open/yolo/valid/"
    # PATH_VAL_LABELS = "../open/yolo/valid/"
    # execute(PATH_TRAIN_IMAGES, PATH_TRAIN_LABELS, "Test", class_labels, run)
    # execute(PATH_VAL_IMAGES, PATH_VAL_LABELS, "Validation", class_labels, run)
    
    #model = YOLO(f"{MODEL}/train/weights/last.pt")
    model = YOLO(args.model)
    #add_wandb_callbacks(model, project=args.wandb_project)
    results = model.train(
        data="../open/yolo/custom.yaml",
        imgsz=(args.imgsz_w, args.imgsz_h),
        save=True,
        save_period=10,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        exist_ok=args.exist_ok,
        project=args.project_name,
        #name=args.wandb_name,
        name=args.model,
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