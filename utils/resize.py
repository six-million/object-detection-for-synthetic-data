import os
import cv2
import glob
from tqdm import tqdm
import albumentations as A
from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser(description='resize')

    parser.add_argument('--max_size', type=int, default=1024)
    parser.add_argument('--base_dir', type=str, default='./open/train') # resize할 이미지들이 있는 폴더 경로
    parser.add_argument('--new_dir', type=str, default='./new_open/train') # resize된 이미지들을 저장할 폴더 경로(없으면 자동 생성)
    
    return parser.parse_args()


def get_transforms(args, mode):
    if mode == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=args.max_size, interpolation=1, always_apply=True, p=1),],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=args.max_size, interpolation=1, always_apply=True, p=1),],)


def check(args):
    img_dirs = sorted(glob.glob(args.base_dir + '/**.png'))
    txt_dirs = sorted(glob.glob(args.base_dir + '/**.txt'))

    if len(txt_dirs) == 0:
        print('test 폴더로 txt 파일 없이 이미지만 resize합니다.')
        return 'test'
    
    for img_dir, txt_dir in zip(img_dirs, txt_dirs):
        if img_dir.replace('.png', '.txt') != txt_dir:
            print(f'{img_dir}에 txt 파일이 없습니다.')
            return None
        
    print('train 폴더로 txt 파일과 이미지 모두 resize합니다.')
    return 'train'


def main(img_dir, mode, args):
    for img_dir in tqdm(img_dirs, total=len(img_dirs)):
        # 이미지 읽어오기
        image = cv2.imread(img_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mode == 'train':
            # txt 파일 경로 확인
            txt_dir = img_dir.replace('.png', '.txt')
            assert os.path.exists(txt_dir), f'{txt_dir} does not exist'

            # annotation 읽어오기
            anno_file = open(txt_dir, 'r')
            anno_lines = [list(map(float, line.strip().split())) for line in anno_file.readlines()]
            anno_file.close()
            
            # bbox와 category_id 저장
            bboxes = []
            category_ids = []
            for line in anno_lines:
                cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line)
                bboxes.append([x1, y1, x3, y3])
                category_ids.append(cls)

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        
        else:
            transformed = transform(image=image)

        # 이미지 저장
        new_img_dir = os.path.join(args.new_dir, img_dir.split('/')[-1])
        cv2.imwrite(new_img_dir, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
        
        if mode == 'train':
            # 주석 저장
            new_txt_dir = new_img_dir.replace('.png', '.txt')
            anno_file = open(new_txt_dir, 'w')
            for bbox, category_id in zip(transformed['bboxes'], transformed['category_ids']):
                xmin, ymin, xmax, ymax = map(int, bbox)
                anno_file.write(f'{float(category_id)} {xmin} {ymin} {xmax} {ymin} {xmax} {ymax} {xmin} {ymax}\n')
            anno_file.close()


if __name__ == '__main__':
    args = parse_args()

    mode = check(args)
    if mode is None:
        exit()

    transform = get_transforms(args, mode)

    img_dirs = sorted(glob.glob(args.base_dir + '/**.png'))

    if not os.path.exists(args.new_dir):
        os.makedirs(args.new_dir)

    main(img_dirs, mode, args)
    