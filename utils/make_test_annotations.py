import json
import os
from PIL import Image

with open('train.json', 'r') as f:  # categories 불러올 json file
    json_data = json.load(f)


new = dict()
new['categories'] = json_data['categories']
new['images'] = []

data_path = 'test/'  # test image file이 있는 경리
id = 1
# images
for file in sorted(os.listdir(data_path)):
    if '.png' in file:
        image_size = Image.open(data_path + file).size
        image = {'file_name': file,
                 'width': image_size[0],
                 'height': image_size[1],
                 'id': id}
        new['images'].append(image)
        id += 1

save_path = './'  # test annotation file이 저장될 경로
with open(save_path + 'test.json', 'w') as f:
    json.dump(new, f, indent = 4)
