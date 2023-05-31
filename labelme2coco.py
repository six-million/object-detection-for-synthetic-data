import os
import json
from PIL import Image

images, annotations, categories = [], [], []
data_path = './'


# images
for file in sorted(os.listdir(data_path)):
    if '.png' in file:
        image_size = Image.open(data_path + file).size
        image = {'file_name': file,
                 'width': image_size[0],
                 'height': image_size[1],
                 'id': int(file[4:9])}
        images.append(image)


# annotations
index = 1
for file in sorted(os.listdir(data_path)):
    if '.txt' in file and 'syn' in file:
        f = open(data_path + file, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            tmp = line.split()

            info = dict()
            info['id'] = index
            info['image_id'] = int(file[4:9])
            info['bbox'] = [float(tmp[1]), float(tmp[2]), float(tmp[3]) - float(tmp[1]), float(tmp[6]) - float(tmp[2])]
            info['area'] = float(tmp[6]) * float(tmp[2])
            info['category_id'] = int(float(tmp[0]))
            annotations.append(info)
            index += 1

        f.close()


# categories
classes_path = './'
f = open(classes_path + 'classes.txt', 'r')
while True:
    line = f.readline()
    if not line:
        break
    tmp = line.split(',')

    info = dict()
    info['id'] = int(tmp[0])
    info['name'] = tmp[1].replace('\n', '')
    categories.append(info)
f.close()


# dict to json
coco_format = dict()
coco_format['images'] = images
coco_format['categories'] = categories
coco_format['annotations'] = annotations

save_path = './'
with open(save_path + 'annotations.json', 'w') as f:
    json.dump(coco_format, f, indent = 4)
