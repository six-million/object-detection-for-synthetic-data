import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

from ensemble_boxes import *
from pycocotools.coco import COCO


def parse_args():
    parser = ArgumentParser(description='ensemble')
    
    parser.add_argument('--base_dir', type=str, default='./open/train')
    parser.add_argument('--test_dir', type=str, default='./open/test.json')
    parser.add_argument('--pred_names', type=list, default=[
        'test_fold0_epoch260.csv',
        'test_fold0_epoch280.csv',
        'test_fold1_epoch240.csv',
        'test_fold1_epoch250.csv',
    ])
    parser.add_argument('--weights', type=list, default=[
        2,
        1,
        3,
        2,
    ])
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--skip_box_thr', type=float, default=0.0001)

    return parser.parse_args()


def get_pred_dict(csv_dir):
    pred_df = pd.read_csv(csv_dir).sort_values(
                                    by=['file_name', 'confidence'],
                                    ascending=[True, False]
                                    ).reset_index(drop=True)

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


def main(args):
    preds = [
        'atss.csv',
        'epoch200_original_size.csv',
    ]
    args.base_dir = './results'
    args.test_dir = './open/test.json'
    #weights=None
    weights = [1, 1]
    iou_thr=0.5
    skip_box_thr=0.0001


    preds = [get_pred_dict(os.path.join(args.base_dir, pred)) for pred in args.pred_names]
    result_dict = defaultdict(list)

    coco = COCO(args.test_dir)
    image_ids = coco.getImgIds()
    # 이미지별 weight box fusion 계산
    for image_id in tqdm(image_ids, total=len(image_ids)):
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        
        bboxes_list = []
        scores_list = []
        labels_list = []
        for pred in preds:
            bboxes = []
            scores = []
            labels = []
            for info in pred[file_name]:
                class_id, confidence, xmin, ymin, _, _, xmax, ymax, _, _ = info
                xmin /= width
                ymin /= height
                xmax /= width
                ymax /= height
                bboxes.append([xmin, ymin, xmax, ymax])
                scores.append(confidence)
                labels.append(class_id)
            
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        result_bboxes, result_scores, result_labels = weighted_boxes_fusion(
            bboxes_list,
            scores_list,
            labels_list,
            weights=args.weights,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
        )
        
        for result_bbox, result_score, result_label in zip(result_bboxes, result_scores, result_labels):
            xmin, ymin, xmax, ymax = result_bbox
            xmin *= width
            ymin *= height
            xmax *= width
            ymax *= height
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            confidence = round(result_score, 5)
            class_id = int(result_label)

            result_dict[file_name].append([class_id, confidence, xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])

    # submission.csv 생성
    # result_df로 변환
    result_df = defaultdict(list)
    for file_name, infos in result_dict.items():
        for info in infos:
            class_id, confidence, xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax = info
            result_df['file_name'].append(file_name)
            result_df['class_id'].append(class_id)
            result_df['confidence'].append(confidence)
            result_df['point1_x'].append(xmin)
            result_df['point1_y'].append(ymin)
            result_df['point2_x'].append(xmax)
            result_df['point2_y'].append(ymin)
            result_df['point3_x'].append(xmax)
            result_df['point3_y'].append(ymax)
            result_df['point4_x'].append(xmin)
            result_df['point4_y'].append(ymax)

    result_df = pd.DataFrame(result_df).sort_values(by=['file_name', 'confidence'], ascending=[True, False]).reset_index(drop=True)
    csv_name = 'ensemble_' + '_'.join([name.replace('.csv','') for name in args.pred_names]) + '_iou_thr_' + str(args.iou_thr) + '_skip_box_thr_' + str(args.skip_box_thr) + '.csv'
    result_df.to_csv(os.path.join(args.base_dir, csv_name), index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)