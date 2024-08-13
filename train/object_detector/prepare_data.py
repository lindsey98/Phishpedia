import shutil
###
# custom_dataset
# |– images
# |   |– image1.jpg
# |   |– image2.jpg
# |   |– …
# |– labels
# |   |– image1.txt
# |   |– image2.txt
# |   |– …
# |– train.txt: List of training
# |– val.txt: List of testing
# |– classes.txt
from pathlib import Path
import os
from tqdm import tqdm

class Annotation:

    @staticmethod
    def _to_yolo_label(labelid, bounds, image_width, image_height):
        x_c = (bounds[0] + bounds[2]/2) / image_width # center
        y_c = (bounds[1] + bounds[3]/2) / image_height # center
        w = (bounds[2]) / image_width
        h = (bounds[3]) / image_height
        return f"{labelid} {x_c} {y_c} {w} {h}"

    @staticmethod
    def write_yolo_labels(file, bboxes, image_width, image_height):
        labels = ""
        for box in bboxes:
            labels = labels + Annotation._to_yolo_label(box['category_id']-1, box['bbox'], image_width, image_height) + "\n" # Note that the cateogry_id startswith 1 so we minus 1
        with open(file, "w") as f:
            f.write(labels)

    @staticmethod
    def find_annotations_for_image(file_name, coco_data):
        image_id = None
        for image in coco_data['images']:
            if image['file_name'].startswith(file_name):
                image_id = image['id']
                image_width = image['width']
                image_height = image['height']
                break

        if image_id is None:
            return None, None, None

        # Now, collect all annotations that reference this image_id
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        return annotations, image_width, image_height

if __name__ == "__main__":
    import json
    label2id = {
        'input': 0,
        'logo': 1
    }

    Data = "./datasets/object_detector_training/"
    img_dir = Path(Data) / "images"
    label_dir = Path(Data) / "labels"

    train_img_dir = Path(img_dir) / 'train'
    val_img_dir = Path(img_dir) / 'val'

    train_label_dir = Path(label_dir) / 'train'
    val_label_dir = Path(label_dir) / 'val'

    os.makedirs(Data, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    orig_data_dir = './datasets/benign_sample_30k'
    with open('./datasets/object_detector_training/coco_train.json', 'r') as json_file:
        orig_train_annotations = json.load(json_file)
    with open('./datasets/object_detector_training/coco_test.json', 'r') as json_file:
        orig_test_annotations = json.load(json_file)

    for dir in tqdm(os.listdir(orig_data_dir)):
        shot_path = Path(orig_data_dir) / dir / 'shot.png'
        if os.path.exists(shot_path):
            ### find the annotation
            annotations, image_width, image_height = Annotation.find_annotations_for_image(dir, orig_train_annotations)
            if annotations:
                ## this is training
                shutil.copyfile(shot_path, Path(train_img_dir) / (str(dir) + '.png'))
                Annotation.write_yolo_labels(
                    Path(train_label_dir) / (str(dir) + '.txt'),
                    annotations,
                    image_width,
                    image_height)
            else:
                annotations, image_width, image_height = Annotation.find_annotations_for_image(dir, orig_test_annotations)
                if annotations:
                    shutil.copyfile(shot_path, Path(val_img_dir) / (str(dir) + '.png'))
                    Annotation.write_yolo_labels(
                        Path(val_label_dir) / (str(dir) + '.txt'),
                        annotations,
                        image_width,
                        image_height)
                else:
                    continue


    # for annotator_dir in os.listdir(ground_truth_dir):
    #     file_list = os.listdir(os.path.join(ground_truth_dir, annotator_dir))
    #     sorted_file_list = sorted(file_list, key=lambda s: s.lower())
    #     for ct, app_dir in enumerate(sorted_file_list):
    #         for file in os.listdir(os.path.join(ground_truth_dir, annotator_dir, app_dir)):
    #             if file.endswith('json'):
    #                 json_file_path = os.path.join(ground_truth_dir, annotator_dir, app_dir, file)
    #                 data = json.load(open(json_file_path, encoding="utf-8"))
    #                 ws = [Widget.from_labelme(d, i) for i, d in enumerate(data["shapes"])]
    #                 image_height = data["imageHeight"]
    #                 image_width = data["imageWidth"]
    #                 if ct < 28:  # training
    #                     Annotation.write_yolo_labels(
    #                         os.path.join(train_label_dir, app_dir+'_'+file.replace("json", "txt")),
    #                                      label2id,
    #                                      ws,
    #                                      image_width,
    #                                      image_height)
    #                 else:
    #                     Widget.write_yolo_labels(
    #                         os.path.join(val_label_dir, app_dir + '_' + file.replace("json", "txt")),
    #                         label2id,
    #                         ws,
    #                         image_width,
    #                         image_height)
    #
    #                 if os.path.exists(json_file_path.replace('.json', '.png')):
    #                     image_path = file.replace('.json', '.png')
    #                 elif os.path.exists(json_file_path.replace('.json', '.jpg')):
    #                     image_path = file.replace('.json', '.jpg')
    #                 else:
    #                     image_path = file.replace('.json', '.jpeg')
    #
    #                 if ct < 28:  # training
    #                     shutil.copy(os.path.join(ground_truth_dir, annotator_dir, app_dir, image_path),
    #                                 os.path.join(train_img_dir, app_dir + '_' + image_path))
    #                 else:
    #                     shutil.copy(os.path.join(ground_truth_dir, annotator_dir, app_dir, image_path),
    #                                 os.path.join(val_img_dir, app_dir + '_' + image_path))