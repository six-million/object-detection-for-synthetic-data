import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Confusion Matrix')
    parser.add_argument('--csv_dir', type=str, default='../results/valid_conf25_iou7_angnms_nms.csv')
    parser.add_argument('--base_dir', type=str, default='../open/train')

    return parser.parse_args()


def box_iou_calc(boxes1, boxes2):
    # <https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py>
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD


    def plot(self, file_name='./'):
        names = [
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
        try:
            import seaborn as sns
            # 보기 편한 방향으로 변환
            self.matrix = np.flip(self.matrix, axis=1)

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 12), tight_layout=True)
            sns.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            
            # x축 값
            xticklabels = names + ['background FP']
            xticklabels = xticklabels[::-1] # matrix 방향 flip해서 x축 뒤집기
            xticklabels = xticklabels if labels else "auto"
            # y축 값
            yticklabels = names + ['background FN']
            yticklabels = yticklabels if labels else "auto"

            sns.heatmap(
                array,
                annot=self.num_classes < 30,
                annot_kws={"size": 8},
                cmap='Blues',
                fmt='.2f',
                square=True,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                ).set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')

            if not os.path.exists('./result_analysis'):
                os.mkdir('./result_analysis')
            fig.savefig(Path('./result_analysis') / file_name, dpi=250)

        except Exception as e:
            print(e)
            pass


    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1


    def return_matrix(self):
        return self.matrix


    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))


def get_pred_dict(csv_dir):
    pred_df = pd.read_csv(csv_dir).sort_values(by=['file_name', 'confidence'], ascending=[True, False]).reset_index(drop=True)
    pred_dict = defaultdict(list)
    for i in range(len(pred_df)):
        file_name = pred_df.iloc[i]['file_name']
        class_id = pred_df.iloc[i]['class_id']
        confidence = pred_df.iloc[i]['confidence']
        xmin = pred_df.iloc[i]['point1_x']
        ymin = pred_df.iloc[i]['point1_y']
        xmax = pred_df.iloc[i]['point3_x']
        ymax = pred_df.iloc[i]['point3_y']
        pred_dict[file_name].append([xmin, ymin, xmax, ymax, confidence, class_id])

    return pred_dict


def get_gt_dict(image_names, base_dir):
    gt_dict = defaultdict(list)
    for file_name in image_names:
        gt_dir = os.path.join(base_dir, file_name.replace(".png", ".txt"))
        gt_file = open(gt_dir, 'r')
        gt_infos = [list(map(float, line.strip().split())) for line in gt_file.readlines()]
        gt_file.close()

        for line in gt_infos:
            class_id, xmin, ymin, _, _, xmax, ymax, _, _ = map(int, line)
            gt_dict[file_name].append([class_id, xmin, ymin, xmax, ymax])
    
    return gt_dict


def main(args):
    conf_mat = ConfusionMatrix(num_classes = 34, CONF_THRESHOLD = 0.01, IOU_THRESHOLD = 0.85)

    pred_dict = get_pred_dict(args.csv_dir)
    gt_dict = get_gt_dict(list(set(pd.read_csv(args.csv_dir)['file_name'])), args.base_dir)

    image_names = list(pred_dict.keys())
    pred_list = [pred_dict[image_name] for image_name in image_names]
    gt_list = [gt_dict[image_name] for image_name in image_names]
    
    # confusion matrix 객체에 적용
    for pred, gt in zip(pred_list, gt_list):
        conf_mat.process_batch(np.array(pred), np.array(gt))
    
    # conf_mat 출력
    conf_mat.plot(file_name=args.csv_dir.split('/')[-1].replace('.csv', '_conf_mat.png'))

if __name__ == "__main__":
    args = parse_args()
    main(args)