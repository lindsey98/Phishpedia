import os
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from .utils import brand_converter, resolution_alignment
import matplotlib.pyplot as plt


def l2_norm(x):
    '''L2 Normalization'''
    if len(x.shape):
        x = x.reshape((x.shape[0], -1))
    return F.normalize(x, p=2, dim=1)


def pred_siamese(img, model, imshow=False, title=None, grayscale=False):
    '''
    Inference for a single image
    :param img: image path in str or image in PIL.Image
    :param model: model to make inference
    :param imshow: enable display of image or not
    :param title: title of displayed image
    :param grayscale: convert image to grayscale or not
    :return feature embedding of shape (2048,)
    '''
    #     img_size = 224
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

    ## Resize the image while keeping the original aspect ratio
    pad_color = 255 if grayscale else (255, 255, 255)
    img = ImageOps.expand(img, (
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=pad_color)

    img = img.resize((img_size, img_size))

    ## Plot the image
    if imshow:
        if grayscale:
            plt.imshow(np.asarray(img), cmap='gray')
        else:
            plt.imshow(np.asarray(img))
        plt.title(title)
        plt.show()

        # Predict the embedding
    with torch.no_grad():
        img = img_transforms(img)
        img = img[None, ...].to(device)
        #         logo_feat = model.features(img).squeeze(-1).squeeze(-1)
        logo_feat = model.features(img)
        logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy()  # L2-normalization final shape is (2048,)

    return logo_feat


def siamese_inference(model, domain_map, logo_feat_list, file_name_list, shot_path: str, gt_bbox, t_s, grayscale=False):
    '''
    Return predicted brand for one cropped image
    :param model: model to use
    :param domain_map: brand-domain dictionary
    :param logo_feat_list: reference logo feature embeddings
    :param file_name_list: reference logo paths
    :param shot_path: path to the screenshot
    :param gt_bbox: 1x4 np.ndarray/list/tensor bounding box coords
    :param t_s: similarity threshold for siamese
    :param grayscale: convert image(cropped) to grayscale or not
    :return: predicted target, predicted target's domain
    '''

    try:
        img = Image.open(shot_path)
    except OSError:  # if the image cannot be identified, return nothing
        print('Screenshot cannot be open')
        return None, None, None

    ## get predicted box --> crop from screenshot
    cropped = img.crop((gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
    img_feat = pred_siamese(cropped, model, imshow=False, title='Original rcnn box', grayscale=grayscale)

    ## get cosine similarity with every protected logo
    sim_list = logo_feat_list @ img_feat.T  # take dot product for every pair of embeddings (Cosine Similarity)
    pred_brand_list = file_name_list

    assert len(sim_list) == len(pred_brand_list)

    ## get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    pred_brand_list = np.array(pred_brand_list)[idx]
    sim_list = np.array(sim_list)[idx]

    # top1,2,3 candidate logos
    top3_logolist = [Image.open(x) for x in pred_brand_list]
    top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in pred_brand_list]
    top3_domainlist = [domain_map[x] for x in top3_brandlist]
    top3_simlist = sim_list

    for j in range(3):
        predicted_brand, predicted_domain = None, None

        ## If we are trying those lower rank logo, the predicted brand of them should be the same as top1 logo, otherwise might be false positive
        if top3_brandlist[j] != top3_brandlist[0]:
            continue

        ## If the largest similarity exceeds threshold
        if top3_simlist[j] >= t_s:
            predicted_brand = top3_brandlist[j]
            predicted_domain = top3_domainlist[j]
            final_sim = top3_simlist[j]

        ## Else if not exceed, try resolution alignment, see if can improve
        else:
            cropped, candidate_logo = resolution_alignment(cropped, top3_logolist[j])
            img_feat = pred_siamese(cropped, model, imshow=False, title=None, grayscale=grayscale)
            logo_feat = pred_siamese(candidate_logo, model, imshow=False, title=None, grayscale=grayscale)
            final_sim = logo_feat.dot(img_feat)
            if final_sim >= t_s:
                predicted_brand = top3_brandlist[j]
                predicted_domain = top3_domainlist[j]
            else:
                break  # no hope, do not try other lower rank logos

        ## If there is a prediction, do aspect ratio check
        if predicted_brand is not None:
            ratio_crop = cropped.size[0] / cropped.size[1]
            ratio_logo = top3_logolist[j].size[0] / top3_logolist[j].size[1]
            # aspect ratios of matched pair must not deviate by more than factor of 2.5
            if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                continue  # did not pass aspect ratio check, try other
            # If pass aspect ratio check, report a match
            else:
                return predicted_brand, predicted_domain, final_sim

    return None, None, top3_simlist[0]

# def siamese_inference_debug(model, domain_map, logo_feat_list, file_name_list, shot_path, gt_bbox, t_s=0.83, grayscale=False):
#     '''
#     Debug version: Return predicted brand for one cropped image
#     :param model: model to use
#     :param domain_map: brand-domain dictionary
#     :param logo_feat_list: reference logo feature embeddings
#     :param file_name_list: reference logo paths
#     :param shot_path: path to the screenshot
#     :param gt_bbox: 1x4 np.ndarray/list/tensor bounding box coords
#     :param t_s: similarity threshold for siamese
#     :param grayscale: convert image(cropped) to grayscale or not
#     :return: predicted target, predicted target's domain
#     '''

#     try:
#         img = Image.open(shot_path)
#     except OSError:  # if the image cannot be identified, return nothing
#         return None, None

#     ## get predicted box --> crop from screenshot
#     cropped = img.crop((gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
#     img_feat = pred_siamese(cropped, model, imshow=False, title='Original rcnn box', grayscale=grayscale)

#     ###### Debug #########################################################################
#     pred_siamese(cropped, model, imshow=True, title='Original rcnn box', grayscale=grayscale)
#     ######################################################################################

#     ## get cosine similarity with every protected logo
#     sim_list = logo_feat_list @ img_feat.T # take dot product for every pair of embeddings (Cosine Similarity)
#     pred_brand_list = file_name_list

#     assert len(sim_list) == len(pred_brand_list)

#     ## get top 10 brands
#     idx = np.argsort(sim_list)[::-1][:10]
#     pred_brand_list = np.array(pred_brand_list)[idx]
#     sim_list = np.array(sim_list)[idx]

#     predicted_brand, predicted_domain = None, None
#     candidate_logo = Image.open(pred_brand_list[0])

#     ###### Debug #########################################################################
#     plt.imshow(candidate_logo)
#     plt.title('Top1 similar logo in targetlist {} Similarity : {:.2f}'.format(brand_converter(pred_brand_list[0].split('/')[-2]), sim_list[0]))
#     plt.show()
#     ######################################################################################

#     ## If the largest similarity exceeds threshold
#     if sim_list[0] >= t_s:

#         predicted_brand = brand_converter(pred_brand_list[0].split('/')[-2])
#         predicted_domain = domain_map[predicted_brand]
#         final_sim = max(sim_list)

#     ## Else if not exeed, try resolution alignment, see if can improve
#     else:
#         cropped, candidate_logo = resolution_alignment(cropped, candidate_logo)
#         img_feat = pred_siamese(cropped, model, imshow=False, title=None, grayscale=grayscale)
#         logo_feat = pred_siamese(candidate_logo, model, imshow=False, title=None, grayscale=grayscale)
#         final_sim = logo_feat.dot(img_feat)
#         if final_sim >= t_s:
#             predicted_brand = brand_converter(pred_brand_list[0].split('/')[-2])
#             predicted_domain = domain_map[predicted_brand]

#             ############ Debug ##############################################################
#             print("Pass resolution alignment")
#             ######################################################################################
#         ############### Debug ################################################################
#         else:
#             print("Not pass resolution alignment")
#         ######################################################################################

#     ## If no prediction, return None
#     if predicted_brand is None:
#         return None, None

#     ## If there is a prediction, do aspect ratio check
#     else:
#         ratio_crop = cropped.size[0]/cropped.size[1]
#         ratio_logo = candidate_logo.size[0]/candidate_logo.size[1]
#         # aspect ratios of matched pair must not deviate by more than factor of 2
#         if max(ratio_crop, ratio_logo)/min(ratio_crop, ratio_logo) > 2:
#             ############# Debug #################################################################
#             print("Not pass aspect ratio check")
#             ######################################################################################
#             return None, None

#         # If pass aspect ratio check, report a match
#         else:
#             ############# Debug ################################################################
#             print("Pass aspect ratio check")
#             ######################################################################################
#             return predicted_brand, predicted_domain



