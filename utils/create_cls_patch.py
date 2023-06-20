import os
import json
from PIL import Image
from tqdm import tqdm

img_path = '/opt/ml/object-detection-for-synthetic-data/open/'

data_path = 'train/'
new_train_path = 'new_train/'
new_val_path = 'new_val/'
train_json_path = '/opt/ml/object-detection-for-synthetic-data/open/5fold/train0.json'
val_json_path = '/opt/ml/object-detection-for-synthetic-data/open/5fold/valid0.json'

if not os.path.exists(img_path + new_train_path):
    os.makedirs(img_path + new_train_path)
if not os.path.exists(img_path + new_val_path):
    os.makedirs(img_path + new_val_path)

with open(train_json_path) as f:
    train_json = json.load(f)
with open(val_json_path) as f:
    val_json = json.load(f)

def mk(path, json_file, new_file_name):
    i = 0
    f = open(f'{img_path}{path}{new_file_name}_label.txt', 'w')
    for x in tqdm(json_file['images']):
        file_name = x['file_name'].split('.')[0]

        img = Image.open(img_path + data_path + file_name + '.png')
        label = open(img_path + data_path + file_name + '.txt', 'r')
        
        lines = label.readlines()
        for line in lines:
            line = line.split()
            category = int(float(line[0]))
            lt = list(map(int, [line[1], line[2]]))
            rb = list(map(int, [line[5], line[6]]))
            img_cropped = img.crop(lt + rb)

            img_cropped.save(f'{img_path}{path}{new_file_name}_{i}.png', 'png')
            f.write(f'{new_file_name}_{i}.png,{category}\n')
            i += 1
        
        label.close()
    f.close()

# mk(new_train_path, train_json, 'train')
# mk(new_val_path, val_json, 'val')
mk(new_test_path, test_json, 'test')
