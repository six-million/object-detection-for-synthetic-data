import csv
import os


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

data = []
csv_path = '/opt/ml/0615.csv'
f = open(csv_path)
csv_data = csv.reader(f)
for row in csv_data:
    if '.png' not in row[0]:
        data.append(row)
        continue
    
    # 좌상단, 우하단 박스 제거
    if int(row[3]) <= 20:
        continue
    if int(row[5]) >= 1900:
        continue

    # # 비율, 면적 구하기
    # index = int(row[1])
    # width = int(row[5]) - int(row[3])
    # height = int(row[8]) - int(row[4])
    # ratio = height / width
    # area = width * height

    # # 비율 범위에 있지 않은 박스 제거
    # if ratio < bbox_ratio[index][0] or ratio > bbox_ratio[index][1]:
    #     continue
    
    # # 면적 범위에 있지 않은 박스 제거
    # if area < bbox_area[index][0] or area > bbox_area[index][1]:
    #     continue

    # # confidence score < 0.2 박스 제거
    # if float(row[2]) < 0.2:
    #     continue


    data.append(row)


f.close()

with open('/opt/ml/abc.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(data)
