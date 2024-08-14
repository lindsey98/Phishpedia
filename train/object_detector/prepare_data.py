import shutil
from pathlib import Path
import os
from tqdm import tqdm
import json

class Annotation:

    @staticmethod
    def _to_yolo_label(labelid, bounds, image_width, image_height):
        """
        Convert bounding box information to YOLO format.

        :param labelid: ID of the label (zero-based)
        :param bounds: Bounding box coordinates (x_min, y_min, width, height)
        :param image_width: Width of the image
        :param image_height: Height of the image
        :return: A string formatted in YOLO label format
        """
        x_c = (bounds[0] + bounds[2] / 2) / image_width  # x center
        y_c = (bounds[1] + bounds[3] / 2) / image_height  # y center
        w = bounds[2] / image_width
        h = bounds[3] / image_height
        return f"{labelid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def write_yolo_labels(file, bboxes, image_width, image_height):
        """
        Write bounding box annotations to a file in YOLO format.

        :param file: Path to the label file
        :param bboxes: List of bounding box dictionaries
        :param image_width: Width of the image
        :param image_height: Height of the image
        """
        labels = "\n".join(
            Annotation._to_yolo_label(
                box['category_id'] - 1, box['bbox'], image_width, image_height
            ) for box in bboxes
        )
        with open(file, "w") as f:
            f.write(labels + "\n")

    @staticmethod
    def find_annotations_for_image(file_name, coco_data):
        """
        Find annotations for a given image in the COCO dataset.

        :param file_name: The name of the image file
        :param coco_data: COCO dataset as a dictionary
        :return: Tuple of annotations list, image width, and image height
        """
        for image in coco_data['images']:
            if image['file_name'].startswith(file_name):
                image_id = image['id']
                annotations = [
                    anno for anno in coco_data['annotations'] if anno['image_id'] == image_id
                ]
                return annotations, image['width'], image['height']

        return None, None, None


def setup_directories(base_dir, sub_dirs):
    """
    Create directories if they don't exist.

    :param base_dir: Base directory path
    :param sub_dirs: List of subdirectories to create under the base directory
    :return: A dictionary mapping subdirectory names to their Path objects
    """
    paths = {}
    for sub_dir in sub_dirs:
        path = Path(base_dir) / sub_dir
        os.makedirs(path, exist_ok=True)
        paths[sub_dir] = path
    return paths

if __name__ == "__main__":
    Data = "./datasets/object_detector_training/"
    orig_data_dir = './datasets/benign_sample_30k'

    paths = setup_directories(Data, ["images/train", "images/val", "labels/train", "labels/val"])

    with open(Path(Data) / 'coco_train.json', 'r') as json_file:
        orig_train_annotations = json.load(json_file)
    with open(Path(Data) / 'coco_test.json', 'r') as json_file:
        orig_test_annotations = json.load(json_file)

    for dir in tqdm(os.listdir(orig_data_dir)):
        shot_path = Path(orig_data_dir) / dir / 'shot.png'
        if not shot_path.exists():
            continue

        # Find annotations in training dataset
        annotations, image_width, image_height = Annotation.find_annotations_for_image(dir, orig_train_annotations)
        if annotations:
            target_img_dir = paths['images/train'] / f"{dir}.png"
            target_label_dir = paths['labels/train'] / f"{dir}.txt"
        else:
            # If not found in training, look in validation dataset
            annotations, image_width, image_height = Annotation.find_annotations_for_image(dir,
                                                                                           orig_test_annotations)
            if not annotations:
                continue  # Skip if no annotations found in either dataset
            target_img_dir = paths['images/val'] / f"{dir}.png"
            target_label_dir = paths['labels/val'] / f"{dir}.txt"

        # Copy image and write labels
        shutil.copyfile(shot_path, target_img_dir)
        Annotation.write_yolo_labels(target_label_dir, annotations, image_width, image_height)
