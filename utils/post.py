import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='post processing')
    parser.add_argument('--csv_dir', type=str, default='./results/epoch_original_size.csv')
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--conf_thr', type=float, default=0.6)
    parser.add_argument('--ignore_class', type=bool, default=True) # 클래스 상관없이 iou_thr 이상인 bbox 제거

    return parser.parse_args()

def get_dict(csv_dir):
    pred_df = pd.read_csv(csv_dir)

    pred_results = defaultdict(list)
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
        pred_results[file_name].append([class_id, confidence, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y])

    return pred_results


def iou(bbox1, bbox2):
    _, _, xmin1, ymin1, _, _, xmax1, ymax1, _, _ = bbox1
    _, _, xmin2, ymin2, _, _, xmax2, ymax2, _, _ = bbox2

    # intersection
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)

    # union
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = area1 + area2 - intersection

    return intersection / union


def nms(bboxes, iou_thr, conf_thr, ignore_class=True):
    # confidence thr 이하인 bbox 제거
    bboxes = [bbox for bbox in bboxes if bbox[1] > conf_thr]
    # confidence 오른순으로 정렬(뒤에서부터 pop하기 위해)
    bboxes = sorted(bboxes, key=lambda x: x[1])

    # nms 적용
    result = []
    while bboxes:
        bbox = bboxes.pop()
        if ignore_class:
            bboxes = [b for b in bboxes if iou(bbox, b) < iou_thr]
        else:
            bboxes = [b for b in bboxes if bbox[0] != b[0] or (iou(bbox, b) < iou_thr and bbox[0] == b[0])]
        result.append(bbox)

    # 다시 confidence 내림순으로 정렬
    result = sorted(result, key=lambda x: x[1], reverse=True)
    
    return result

def main(pred_results, args):
    # nms 적용
    for k,v in pred_results.items():
        pred_results[k] = nms(v, iou_thr=args.iou_thr, conf_thr=args.conf_thr, ignore_class=args.ignore_class)

    # df로 변환
    result_df = defaultdict(list)
    for k,v in pred_results.items():
        for bbox in v:
            result_df['file_name'].append(k)
            result_df['class_id'].append(bbox[0])
            result_df['confidence'].append(bbox[1])
            result_df['point1_x'].append(bbox[2])
            result_df['point1_y'].append(bbox[3])
            result_df['point2_x'].append(bbox[4])
            result_df['point2_y'].append(bbox[5])
            result_df['point3_x'].append(bbox[6])
            result_df['point3_y'].append(bbox[7])
            result_df['point4_x'].append(bbox[8])
            result_df['point4_y'].append(bbox[9])
    result_df = pd.DataFrame(result_df).sort_values(by=['file_name', 'confidence'], ascending=[True, False]).reset_index(drop=True)
    
    # 저장
    result_df.to_csv(args.csv_dir.replace('.csv', '_nms.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    pred_results = get_dict(args.csv_dir)
    main(pred_results, args)
    