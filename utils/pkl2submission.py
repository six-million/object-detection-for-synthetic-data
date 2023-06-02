import argparse
import pickle
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='pkl file to submission csv file')
    parser.add_argument('--pkl_path')
    parser.add_argument('--csv_path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    pkl_path = args.pkl_path

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    new = []
    for result in tqdm(results):
        file_name = result['img_path'].split('/')[-1]
        
        for i in range(len(result['pred_instances']['bboxes'])):
            bbox = result['pred_instances']['bboxes'][i]
            point1_x, point1_y = bbox[0], bbox[1]
            point2_x, point2_y = bbox[2], bbox[1]
            point3_x, point3_y = bbox[2], bbox[3]
            point4_x, point4_y = bbox[0], bbox[3]
            class_id = result['pred_instances']['labels'][i]
            confidence = result['pred_instances']['scores'][i]

            row = [file_name, class_id.item(), confidence.item(),
                point1_x.item(), point1_y.item(), point2_x.item(), point2_y.item(),
                point3_x.item(), point3_y.item(), point4_x.item(), point4_y.item()]
            new.append(row)

        new_df = pd.DataFrame(new)
        new_df.columns = ['file_name', 'class_id', 'confidence',
                        'point1_x', 'point1_y', 'point2_x', 'point2_y',
                        'point3_x', 'point3_y', 'point4_x', 'point4_y']

    csv_path = args.csv_path
    new_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
