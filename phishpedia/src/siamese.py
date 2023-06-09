from .siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from .siamese_pedia.utils import brand_converter
from .siamese_pedia.inference import siamese_inference, pred_siamese
import torch
import os
import numpy as np
from collections import OrderedDict
import pickle
from tqdm import tqdm
import tldextract

def phishpedia_config(num_classes:int, weights_path:str, targetlist_path:str, grayscale=False):
    '''
    Load phishpedia configurations
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :param targetlist_path: targetlist folder
    :param grayscale: convert logo to grayscale or not, default is RGB
    :return model: siamese model
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.split('module.')[1]
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

#     Prediction for targetlists
    logo_feat_list = []
    file_name_list = []
    
    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'): # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith('.jpg') or logo_path.endswith('.PNG') \
                                          or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'): # skip homepage/loginpage
                    continue
                logo_feat_list.append(pred_siamese(img=os.path.join(targetlist_path, target, logo_path), 
                                                   model=model, grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))
        
    return model, np.asarray(logo_feat_list), np.asarray(file_name_list)


def phishpedia_config_easy(num_classes: int, weights_path: str):
    '''
    Load phishpedia model only
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :param targetlist_path: targetlist folder
    :param grayscale: convert logo to grayscale or not, default is RGB
    :return model: siamese model
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.split('module.')[1]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


def phishpedia_classifier_logo(logo_boxes,
                          domain_map_path: str,
                          model, logo_feat_list, file_name_list, shot_path: str,
                          url: str,
                          ts: float):
    '''
    Run siamese
    :param logo_boxes: torch.Tensor/np.ndarray Nx4 logo box coords
    :param domain_map_path: path to domain map dict
    :param model: siamese model
    :param logo_feat_list: targetlist embeddings
    :param file_name_list: targetlist paths
    :param shot_path: path to image
    :param url: url
    :param ts: siamese threshold
    :return pred_target
    :return coord: coordinate for matched logo
    '''
    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    print('number of logo boxes:', len(logo_boxes))
    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None

    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = siamese_inference(model, domain_map,
                                                                    logo_feat_list, file_name_list,
                                                                    shot_path, bbox, t_s=ts, grayscale=False)
            # print(target_this, domain_this, this_conf)
            # domain matcher to avoid FP
            if matched_target is not None:
                if tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix not in matched_domain:
                    # avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                    if matched_target == 'GoDaddy' or matched_target == "Webmail Provider" or matched_target == "Government of the United Kingdom":
                        matched_target = None  # ignore the prediction
                        matched_domain = None  # ignore the prediction
                        this_conf = None
                else:  # benign, real target
                    matched_target = None  # ignore the prediction
                    matched_domain = None  # ignore the prediction
                    this_conf = None
                break  # break if target is matched
            if i >= 2:  # only look at top-2 logo
                break

    return brand_converter(matched_target), matched_coord, this_conf

