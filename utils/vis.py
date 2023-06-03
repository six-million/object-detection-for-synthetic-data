import os
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict

from map import do_voc_evaluation


def parse_args():
    parser = ArgumentParser(description='post processing')
    parser.add_argument('--csv_dir', type=str, default='./results/valid_conf25_iou7_angnms_nms.csv')
    parser.add_argument('--image_num', type=int, default=40) # 출력할 이미지 개수: 40 넘어가면 한 장에 출력 불가 오류 발생
    parser.add_argument('--base_dir', type=str, default='../open/train')

    return parser.parse_args()


# 클래스 - encoding 변환 함수
# def get_convert(class_dir='./open/classes.txt'):
#     f = open(class_dir, 'r')
#     lines = [list(line.strip().split(',')) for line in f.readlines()]
#     f.close()

#     encode = {}
#     decode = {}
#     for cls_num, cls_name in lines:
#         encode[int(cls_num)] = cls_name
#         decode[cls_name] = int(cls_num)
    
#     return encode, decode
# encode, decode = get_convert()
encode = {
    0: "chevrolet_malibu_sedan_2012_2016",
    1: "chevrolet_malibu_sedan_2017_2019",
    2: "chevrolet_spark_hatchback_2016_2021",
    3: "chevrolet_trailblazer_suv_2021_",
    4: "chevrolet_trax_suv_2017_2019",
    5: "genesis_g80_sedan_2016_2020",
    6: "genesis_g80_sedan_2021_",
    7: "genesis_gv80_suv_2020_",
    8: "hyundai_avante_sedan_2011_2015",
    9: "hyundai_avante_sedan_2020_",
    10: "hyundai_grandeur_sedan_2011_2016",
    11: "hyundai_grandstarex_van_2018_2020",
    12: "hyundai_ioniq_hatchback_2016_2019",
    13: "hyundai_sonata_sedan_2004_2009",
    14: "hyundai_sonata_sedan_2010_2014",
    15: "hyundai_sonata_sedan_2019_2020",
    16: "kia_carnival_van_2015_2020",
    17: "kia_carnival_van_2021_",
    18: "kia_k5_sedan_2010_2015",
    19: "kia_k5_sedan_2020_",
    20: "kia_k7_sedan_2016_2020",
    21: "kia_mohave_suv_2020_",
    22: "kia_morning_hatchback_2004_2010",
    23: "kia_morning_hatchback_2011_2016",
    24: "kia_ray_hatchback_2012_2017",
    25: "kia_sorrento_suv_2015_2019",
    26: "kia_sorrento_suv_2020_",
    27: "kia_soul_suv_2014_2018",
    28: "kia_sportage_suv_2016_2020",
    29: "kia_stonic_suv_2017_2019",
    30: "renault_sm3_sedan_2015_2018",
    31: "renault_xm3_suv_2020_",
    32: "ssangyong_korando_suv_2019_2020",
    33: "ssangyong_tivoli_suv_2016_2020",
}


# 클래스 - 색상 변환
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
         (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), 
         (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
         (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0)]


def get_pred_df(csv_dir):
    return pd.read_csv(csv_dir).sort_values(by=['file_name', 'confidence'], ascending=[True, False]).reset_index(drop=True)


def get_gt_df(image_names, base_dir):
    gt_results = defaultdict(list)

    for image_name in image_names:
        gt_dir = os.path.join(base_dir, image_name.replace(".png", ".txt"))
        #gt_dir = f'./open/train/{image_name.replace(".png", ".txt")}'
        gt_file = open(gt_dir, 'r')
        gt_infos = [list(map(float, line.strip().split())) for line in gt_file.readlines()]
        gt_file.close()
        for line in gt_infos:
            cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line)
        
            gt_results['file_name'].append(image_name)
            gt_results['class_id'].append(cls)
            gt_results['point1_x'].append(x1)
            gt_results['point1_y'].append(y1)
            gt_results['point2_x'].append(x2)
            gt_results['point2_y'].append(y2)
            gt_results['point3_x'].append(x3)
            gt_results['point3_y'].append(y3)
            gt_results['point4_x'].append(x4)
            gt_results['point4_y'].append(y4)
    
    return pd.DataFrame(gt_results)


def get_pred_dict(csv_dir):
    pred_df = pd.read_csv(csv_dir).sort_values(by=['file_name', 'confidence'], ascending=[True, False]).reset_index(drop=True)

    pred_dict = defaultdict(list)
    for i in range(len(pred_df)):
        file_name = pred_df.iloc[i]['file_name']
        class_id = pred_df.iloc[i]['class_id']
        confidence = pred_df.iloc[i]['confidence']
        point1_x = pred_df.iloc[i]['point1_x']
        point1_y = pred_df.iloc[i]['point1_y']
        point2_x = pred_df.iloc[i]['point2_x']
        point2_y = pred_df.iloc[i]['point2_y']
        point3_x = pred_df.iloc[i]['point3_x']
        point3_y = pred_df.iloc[i]['point3_y']
        point4_x = pred_df.iloc[i]['point4_x']
        point4_y = pred_df.iloc[i]['point4_y']
        pred_dict[file_name].append([class_id, confidence, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y])

    return pred_dict


def get_gt_dict(image_names, base_dir):
    gt_dict = defaultdict(list)

    for file_name in image_names:
        gt_dir = os.path.join(base_dir, file_name.replace(".png", ".txt"))
        #gt_dir = f'./open/train/{file_name.replace(".png", ".txt")}'
        gt_file = open(gt_dir, 'r')
        gt_infos = [list(map(float, line.strip().split())) for line in gt_file.readlines()]
        gt_file.close()

        for line in gt_infos:
            class_id, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y = map(int, line)

            gt_dict[file_name].append([class_id, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y])
    
    return gt_dict


def get_map_df(pred_dict, gt_dict):
    image_names = sorted(list(set(pred_dict.keys())))

    map_dict = defaultdict(int)
    for image_name in image_names:
        preds = defaultdict(list)
        for pred in pred_dict[image_name]:
            preds['file_name'].append(image_name)
            preds['class_id'].append(pred[0])
            preds['confidence'].append(pred[1])
            preds['point1_x'].append(pred[2])
            preds['point1_y'].append(pred[3])
            preds['point2_x'].append(pred[4])
            preds['point2_y'].append(pred[5])
            preds['point3_x'].append(pred[6])
            preds['point3_y'].append(pred[7])
            preds['point4_x'].append(pred[8])
            preds['point4_y'].append(pred[9])
        preds = pd.DataFrame(preds)

        gts = defaultdict(list)
        for gt in gt_dict[image_name]:
            gts['file_name'].append(image_name)
            gts['class_id'].append(gt[0])
            gts['point1_x'].append(gt[1])
            gts['point1_y'].append(gt[2])
            gts['point2_x'].append(gt[3])
            gts['point2_y'].append(gt[4])
            gts['point3_x'].append(gt[5])
            gts['point3_y'].append(gt[6])
            gts['point4_x'].append(gt[7])
            gts['point4_y'].append(gt[8])
        gts = pd.DataFrame(gts)

        map_dict[image_name] = do_voc_evaluation(gts, preds)

    return map_dict


def put_bboxes(image_dir, bboxes, withconf=True):
    image = cv2.imread(image_dir)
    for bbox in bboxes:
        if withconf:
            class_id, confidence, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y = bbox
            cv2.rectangle(image, (point1_x, point1_y), (point3_x, point3_y), palette[class_id], 2)
            cv2.putText(image, f'{encode[class_id]}', (point1_x, point1_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, palette[class_id], 3)
            cv2.putText(image, f'{confidence:.2f}', (point4_x, point4_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, palette[class_id], 3)
            cv2.putText(image, f'{(point3_x-point1_x)*(point3_y-point1_y)}', (point3_x, point3_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, palette[class_id], 3)
        else:
            class_id, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y = bbox
            cv2.rectangle(image, (point1_x, point1_y), (point3_x, point3_y), palette[class_id], 2)
            cv2.putText(image, f'{encode[class_id]}', (point1_x, point1_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, palette[class_id], 3)
            cv2.putText(image, f'{(point3_x-point1_x)*(point3_y-point1_y)}', (point3_x, point3_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, palette[class_id], 3)
    
    return cv2.resize(image, (512, 277))


def save_images(map_results, pred_dict, gt_dict, base_dir, save_dir, mAP):
    num = len(map_results)
    fig, axes = plt.subplots(num, 2, figsize=(30, 15*num))

    for i, (image_name, map_score) in tqdm(enumerate(map_results), total=num):
        #image_dir = f'./open/train/{image_name}'
        image_dir = os.path.join(base_dir, image_name)
        axes[i, 0].imshow(put_bboxes(image_dir, pred_dict[image_name], withconf=True))
        axes[i, 0].set_title(f'mAP85: {map_score:.2f} \n pred: {image_name}', fontsize=50)

        axes[i, 1].imshow(put_bboxes(image_dir, gt_dict[image_name], withconf=False))
        axes[i, 1].set_title(f'gt: {image_name}', fontsize=50)

    #fig.suptitle(f'mAP85 {mAP:.2f}', fontsize=50)
    fig.set_tight_layout(True)
    fig.savefig(save_dir)
    print('image saved')


def main(args):
    mAP = do_voc_evaluation(get_gt_df(list(set(pd.read_csv(args.csv_dir)['file_name'])), args.base_dir), get_pred_df(args.csv_dir))
    print(f'total mAP85 = {mAP:.6f}')

    pred_dict = get_pred_dict(args.csv_dir)
    image_names = list(pred_dict.keys())
    gt_dict = get_gt_dict(image_names, args.base_dir)
    map_dict = get_map_df(pred_dict, gt_dict)
    map_results = sorted([[k,v] for k,v in map_dict.items()], key=lambda x: x[1])

    # mAP 분포
    analysis = defaultdict(int)
    for k,v in map_results:
        analysis[v] += 1
    # mAP 분포 출력
    for k, v in analysis.items():
        print(f'mAP85 = {k:.2f}: {v}장')
    

    save_dir = args.csv_dir.replace('.csv', '.png')
    save_images(map_results[:args.image_num], pred_dict, gt_dict, args.base_dir, save_dir, mAP)


if __name__ == '__main__':
    args = parse_args()
    main(args)
