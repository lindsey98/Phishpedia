import logging
from PIL import Image, ImageOps
from torchvision import transforms
from utils import brand_converter, resolution_alignment, l2_norm
import torch
import os
import numpy as np
from tqdm import tqdm
from tldextract import tldextract
import pickle
import onnxruntime
from typing import Dict
from memory_profiler import profile

logging.basicConfig(level=logging.INFO)

def check_domain_brand_inconsistency(logo_boxes, domain_map: Dict, model, logo_feat_list,
                                     file_name_list, shot_path: str, url: str, ts: float, topk: int = 3):

    if tldextract.extract(url).domain == 'google': # ignore Google which has many variants of tlds
        return None, None, None, None

    extracted_domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    logging.info('Number of logo boxes: {}'.format(len(logo_boxes)))

    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None

    if len(logo_boxes) > 0:
        for i, coord in enumerate(logo_boxes[:topk]):  # Process only up to topk elements
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = pred_brand(model, domain_map,
                                                                   logo_feat_list, file_name_list,
                                                                   shot_path, bbox, t_s=ts,
                                                                   do_aspect_ratio_check=False,
                                                                   do_resolution_alignment=False)

            if matched_target and matched_domain:
                matched_coord = coord
                # Check if the domain is part of any domain listed under the brand
                if extracted_domain in matched_domain:
                    matched_target, matched_domain = None, None  # Clear if domains are consistent
                else:
                    break  # Inconsistent domain found, break the loop

    return brand_converter(matched_target), matched_domain, matched_coord, this_conf

def load_model_weights(weights_path: str):
    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(weights_path,
                                           sess_options,
                                           providers=["CPUExecutionProvider"])
    return session

def cache_reference_list(model, targetlist_path: str):
    '''
    cache the embeddings of the reference list
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :param targetlist_path: targetlist folder
    :return model: siamese model
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''

    VALID_EXTENSIONS = {'.png', '.jpeg', '.jpg'}
    SKIP_PREFIXES = {'loginpage', 'homepage'}

    logo_feat_list = []
    file_name_list = []

    for target in tqdm(os.listdir(targetlist_path)):
        target_path = os.path.join(targetlist_path, target)
        if target.startswith('.'):
            continue
        for logo_path in os.listdir(target_path):
            full_path = os.path.join(target_path, logo_path)
            if any(logo_path.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                if any(logo_path.startswith(prefix) for prefix in SKIP_PREFIXES):
                    continue
                try:
                    logo_feat = get_embedding(img=full_path, model=model)
                    logo_feat_list.append(logo_feat)
                    file_name_list.append(full_path)
                except Exception as e:
                    logging.error(f"Error processing {full_path}: {str(e)}")

    logo_feat_arr = np.asarray(logo_feat_list)
    logo_file_arr = np.asarray(file_name_list)
    return logo_feat_arr, logo_file_arr

@torch.inference_mode()
def get_embedding(img, model):
    '''
    Inference for a single image
    :param img: image path in str or image in PIL.Image
    :param model: model to make inference
    :param imshow: enable display of image or not
    :param title: title of displayed image
    :return feature embedding of shape (2048,)
    '''
    # img_size = 64
    img_size = 128

    # Define the image transformation pipeline
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load the image if it's a file path
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")

    ## Resize the image while keeping the original aspect ratio
    img = ImageOps.expand(img, (
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

    img = img.resize((img_size, img_size))

    # Apply transformations and add batch dimension
    img = img_transforms(img).unsqueeze(0)  # (1, 3, 64, 64)

    # Run inference using ONNX model
    ort_outputs = model.run(['features'], {'input': img.numpy()})
    logo_feat = ort_outputs[0]
    logo_feat = logo_feat / np.linalg.norm(logo_feat, ord=2) # l2 normalization
    logo_feat = logo_feat[0] # remove dummy dimension
    return logo_feat

def load_image(image_path):
    try:
        return Image.open(image_path)
    except IOError:
        logging.error(f'Cannot open image: {image_path}')
        return None

def get_top_predictions(similarities, file_names, top_n=3):
    idx = np.argsort(similarities)[::-1][:top_n]
    return np.array(file_names)[idx], np.array(similarities)[idx]

def check_aspect_ratio(cropped, candidate_logo, cap_thre=2.5):
    ratio_crop = cropped.size[0] / cropped.size[1]
    ratio_logo = candidate_logo.size[0] / candidate_logo.size[1]
    return max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) <= cap_thre

def pred_brand(model, domain_map, logo_feat_list, file_name_list, shot_path: str, gt_bbox, t_s,
               do_resolution_alignment=True,
               do_aspect_ratio_check=True):
    '''
    Return predicted brand for one cropped image
    :param model: model to use
    :param domain_map: brand-domain dictionary
    :param logo_feat_list: reference logo feature embeddings
    :param file_name_list: reference logo paths
    :param shot_path: path to the screenshot
    :param gt_bbox: 1x4 np.ndarray/list/tensor bounding box coords
    :param t_s: similarity threshold for siamese
    :param do_resolution_alignment: if the similarity does not exceed the threshold, do we align their resolutions to have a retry
    :param do_aspect_ratio_check: once two logos are similar, whether we want to a further check on their aspect ratios
    :return: predicted target, predicted target's domain
    '''
    cropped_image = load_image(shot_path)
    if not cropped_image:
        return None, None, None

    cropped = cropped_image.crop(gt_bbox)
    img_feat = get_embedding(cropped, model)

    sim_list = logo_feat_list @ img_feat.T  # Cosine Similarity
    pred_brand_list, top_similarities = get_top_predictions(sim_list, file_name_list)

    for j in range(3):
        predicted_brand = os.path.basename(os.path.dirname(str(pred_brand_list[j])))
        predicted_domain = domain_map.get(predicted_brand, None)

        if top_similarities[j] >= t_s:
            return predicted_brand, predicted_domain, top_similarities[j]

    return None, None, top_similarities[0]  # Returning the highest similarity for reference
