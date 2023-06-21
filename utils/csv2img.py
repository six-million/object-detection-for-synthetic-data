import csv
from collections import defaultdict
from tqdm import tqdm
import cv2
import os

data = defaultdict(list)

csv_path = '/opt/ml/0615.csv'
f = open(csv_path)
csv_data = csv.reader(f)
for row in csv_data:
    if '.png' not in row[0]:
        continue
    data[row[0]].append(row[1:])

f.close()

img_path = '/opt/ml/object-detection-for-synthetic-data/open/test/'


def min_max_bbox():
    txt_path = '/opt/ml/object-detection-for-synthetic-data/open/train/'
    txt_list = os.listdir(txt_path)

    bbox_ratio = [[1000000, -1] for _ in range(34)]  # 세로/가로의 최소, 최대
    bbox_area = [[1000000, -1] for _ in range(34)]  # 면적의 최소, 최대

    ratio_pad = 0.8
    area_pad = 0.8

    for txt in txt_list:
        if '.png' in txt:
            continue
        f = open(txt_path + txt, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split()
            index = int(float(line[0]))
            width = int(line[3]) - int(line[1])
            height = (int(line[6]) - int(line[2])) * 1080 / 1040

            ratio = height / width
            area = width * height

            if bbox_ratio[index][0] > ratio * ratio_pad:
                bbox_ratio[index][0] = ratio * ratio_pad
            if bbox_ratio[index][1] < ratio * (2 - ratio_pad):
                bbox_ratio[index][1] = ratio * (2 - ratio_pad)
            
            if bbox_area[index][0] > area * area_pad:
                bbox_area[index][0] = area * area_pad
            if bbox_area[index][1] < area * (2 - area_pad):
                bbox_area[index][1] = area * (2 - area_pad)

        f.close()

    bbox_ratio = {i: bbox_ratio[i] for i in range(34)}
    bbox_area = {i: bbox_area[i] for i in range(34)}

    return bbox_ratio, bbox_area

bbox_ratio, bbox_area = min_max_bbox()

classes = [
'chevrolet_malibu_sedan_2012_2016'
,'chevrolet_malibu_sedan_2017_2019'
,'chevrolet_spark_hatchback_2016_2021'
,'chevrolet_trailblazer_suv_2021_'
,'chevrolet_trax_suv_2017_2019'
,'genesis_g80_sedan_2016_2020'
,'genesis_g80_sedan_2021_'
,'genesis_gv80_suv_2020_'
,'hyundai_avante_sedan_2011_2015'
,'hyundai_avante_sedan_2020_'
,'hyundai_grandeur_sedan_2011_2016'
,'hyundai_grandstarex_van_2018_2020'
,'hyundai_ioniq_hatchback_2016_2019'
,'hyundai_sonata_sedan_2004_2009'
,'hyundai_sonata_sedan_2010_2014'
,'hyundai_sonata_sedan_2019_2020'
,'kia_carnival_van_2015_2020'
,'kia_carnival_van_2021_'
,'kia_k5_sedan_2010_2015'
,'kia_k5_sedan_2020_'
,'kia_k7_sedan_2016_2020'
,'kia_mohave_suv_2020_'
,'kia_morning_hatchback_2004_2010'
,'kia_morning_hatchback_2011_2016'
,'kia_ray_hatchback_2012_2017'
,'kia_sorrento_suv_2015_2019'
,'kia_sorrento_suv_2020_'
,'kia_soul_suv_2014_2018'
,'kia_sportage_suv_2016_2020'
,'kia_stonic_suv_2017_2019'
,'renault_sm3_sedan_2015_2018'
,'renault_xm3_suv_2020_'
,'ssangyong_korando_suv_2019_2020'
,'ssangyong_tivoli_suv_2016_2020'
]

new_img_path = 'ttt/'
import os

del_ltop_rbottom = True
del_conf, conf_thr = False, 0.2
del_bbox_ratio_area = False


if not os.path.exists(new_img_path):
    os.makedirs(new_img_path)

for x in tqdm(data):
    image = cv2.imread(img_path + x)
    for i, bbox in enumerate(data[x]):
        if del_ltop_rbottom:
            # 좌상단 박스 제거
            if int(bbox[2]) <= 20:
                continue
            if int(bbox[4]) >= 1900:
                continue


        cls = classes[int(bbox[0])]
        conf = bbox[1]

        # confidence score 제한 주기
        if del_conf:
            if float(conf) <= conf_thr:
                continue
        
        # 비율과 면적 범위에 들지 않는 박스 제거
        if del_bbox_ratio_area:
            index = int(bbox[0])
            width = int(bbox[4]) - int(bbox[2])
            height = int(bbox[7]) - int(bbox[3])
            ratio = height / width
            area = width * height
            if ratio < bbox_ratio[index][0] or ratio > bbox_ratio[index][1]:
                continue
            if area < bbox_area[index][0] or area > bbox_area[index][1]:
                continue

        cv2.rectangle(image, list(map(int, bbox[2:4])), list(map(int, bbox[6:8])), (0, 255, 0), thickness=4)
        cv2.putText(image, str(i), [int(bbox[2]), int(bbox[3]) + 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, str(i) + '  ' + cls + '  ' + conf, [0, 0 + 30*(i+1)], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(image, conf, [0, 0 + 60*(i+1)], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image = cv2.resize(image, (800, 500))
    cv2.imwrite(new_img_path + x, image)
