from PIL import Image, ImageOps
from torchvision import transforms
from utils import brand_converter, resolution_alignment, l2_norm
from models import KNOWN_MODELS
import torch
import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from tldextract import tldextract
import pickle


def check_domain_brand_inconsistency(logo_boxes,
                                     domain_map_path: str,
                                     model, logo_feat_list,
                                     file_name_list, shot_path: str,
                                     url: str, ts: float,
                                     topk: float=3):

    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    print('number of logo boxes:', len(logo_boxes))
    extracted_domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None

    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):

            if i == topk:
                break

            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = pred_brand(model, domain_map,
                                                                    logo_feat_list, file_name_list,
                                                                    shot_path, bbox, t_s=ts, grayscale=False,
                                                                   do_aspect_ratio_check=False, do_resolution_alignment=False)
            # print(target_this, domain_this, this_conf)
            # domain matcher to avoid FP
            if matched_target and matched_domain:
                matched_coord = coord
                # Check if the domain is part of any domain listed under the brand
                if extracted_domain in matched_domain:
                    matched_target, matched_domain = None, None  # Clear if domains are consistent
                else:
                    break  # Inconsistent domain found, break the loop

    return brand_converter(matched_target), matched_domain, matched_coord, this_conf


def load_model_weights(num_classes: int, weights_path: str):
    '''
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :return model: siamese model
    '''
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if 'module.' in k:
            name = k.split('module.')[1]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def cache_reference_list(model, targetlist_path: str, grayscale=False):
    '''
    cache the embeddings of the reference list
    :param targetlist_path: targetlist folder
    :param grayscale: convert logo to grayscale or not, default is RGB
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''

    # Prediction for targetlists
    logo_feat_list = []
    file_name_list = []
    SUPPORTED_EXTENSIONS = {'.png', '.jpeg', '.jpg', '.PNG', '.JPG', '.JPEG'}
    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'):  # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if any(logo_path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'):  # skip homepage/loginpage
                    continue
                logo_feat_list.append(get_embedding(img=os.path.join(targetlist_path, target, logo_path),
                                                    model=model, grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))

    return np.asarray(logo_feat_list), np.asarray(file_name_list)


@torch.no_grad()
def get_embedding(img, model, grayscale=False):
    '''
    Inference for a single image
    :param img: image path in str or image in PIL.Image
    :param model: model to make inference
    :param imshow: enable display of image or not
    :param title: title of displayed image
    :param grayscale: convert image to grayscale or not
    :return feature embedding of shape (2048,)
    '''
    # img_size = 224
    img_size = 128
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
        ])

    img = Image.open(img) if isinstance(img, str) else img
    img = img.convert("L").convert("RGB") if grayscale else img.convert("RGB")

    # Resize the image while keeping the original aspect ratio
    pad_color = 255 if grayscale else (255, 255, 255)
    img = ImageOps.expand(img, (
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=pad_color)

    img = img.resize((img_size, img_size))

    # Predict the embedding
    img = img_transforms(img)
    img = img[None, ...].to(device)
    logo_feat = model.features(img)
    logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy()  # L2-normalization final shape is (2048,)

    return logo_feat


def pred_brand(model, domain_map, logo_feat_list, file_name_list, shot_path: str, gt_bbox, t_s,
               grayscale=False,
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
    :param grayscale: convert image(cropped) to grayscale or not
    :return: predicted target, predicted target's domain
    '''

    try:
        img = Image.open(shot_path)
    except OSError:  # if the image cannot be identified, return nothing
        print('Screenshot cannot be open')
        return None, None, None

    # get predicted box --> crop from screenshot
    cropped = img.crop((gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
    img_feat = get_embedding(cropped, model, grayscale=grayscale)

    # get cosine similarity with every protected logo
    sim_list = logo_feat_list @ img_feat.T  # take dot product for every pair of embeddings (Cosine Similarity)
    pred_brand_list = file_name_list

    assert len(sim_list) == len(pred_brand_list)

    # get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    pred_brand_list = np.array(pred_brand_list)[idx]
    sim_list = np.array(sim_list)[idx]

    # top1,2,3 candidate logos
    top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in pred_brand_list]
    top3_domainlist = [domain_map[x] for x in top3_brandlist]
    top3_simlist = sim_list

    for j in range(3):
        predicted_brand, predicted_domain = None, None

        # If we are trying those lower rank logo, the predicted brand of them should be the same as top1 logo, otherwise might be false positive
        if top3_brandlist[j] != top3_brandlist[0]:
            continue

        # If the largest similarity exceeds threshold
        if top3_simlist[j] >= t_s:
            predicted_brand = top3_brandlist[j]
            predicted_domain = top3_domainlist[j]
            final_sim = top3_simlist[j]

        # Else if not exceed, try resolution alignment, see if can improve
        elif do_resolution_alignment:
            orig_candidate_logo = Image.open(pred_brand_list[j])
            cropped, candidate_logo = resolution_alignment(cropped, orig_candidate_logo)
            img_feat = get_embedding(cropped, model, grayscale=grayscale)
            logo_feat = get_embedding(candidate_logo, model, grayscale=grayscale)
            final_sim = logo_feat.dot(img_feat)
            if final_sim >= t_s:
                predicted_brand = top3_brandlist[j]
                predicted_domain = top3_domainlist[j]
            else:
                break  # no hope, do not try other lower rank logos

        # If there is a prediction, do aspect ratio check
        if predicted_brand is not None:
            if do_aspect_ratio_check:
                orig_candidate_logo = Image.open(pred_brand_list[j])
                ratio_crop = cropped.size[0] / cropped.size[1]
                ratio_logo = orig_candidate_logo.size[0] / orig_candidate_logo.size[1]
                # aspect ratios of matched pair must not deviate by more than factor of 2.5
                if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                    continue  # did not pass aspect ratio check, try other
            return predicted_brand, predicted_domain, final_sim

    return None, None, top3_simlist[0]
