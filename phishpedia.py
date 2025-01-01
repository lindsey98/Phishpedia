import time
from datetime import datetime
import argparse
import os
import torch
import cv2
from configs import load_config
from logo_recog import pred_rcnn, vis
from logo_matching import check_domain_brand_inconsistency
# from text_recog import check_email_credential_taking
# import pickle
from tqdm import tqdm

import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def result_file_write(f, folder, url, phish_category, pred_target, matched_domain, siamese_conf, logo_recog_time,
                      logo_match_time):
    f.write(folder + "\t")
    f.write(url + "\t")
    f.write(str(phish_category) + "\t")
    f.write(str(pred_target) + "\t")  # write top1 prediction only
    f.write(str(matched_domain) + "\t")
    f.write(str(siamese_conf) + "\t")
    f.write(str(round(logo_recog_time, 4)) + "\t")
    f.write(str(round(logo_match_time, 4)) + "\n")


class PhishpediaWrapper:
    _caller_prefix = "PhishpediaWrapper"
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self._load_config()
        self._to_device()

    def _load_config(self):
        self.ELE_MODEL, self.SIAMESE_THRE, self.SIAMESE_MODEL, \
            self.LOGO_FEATS, self.LOGO_FILES, \
            self.DOMAIN_MAP_PATH = load_config()
        print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    def _to_device(self):
        self.SIAMESE_MODEL.to(self._DEVICE)

    # def simple_input_box_regex(self, html_path):
    #     with open(html_path, 'r', encoding='ISO-8859-1') as f:
    #         page = f.read()
    #         tree = html.fromstring(page)
    #     if tree is None:  # parsing into tree failed
    #         return False

    #     ## filter out search boxes
    #     inputs = tree.xpath(
    #         './/input[not(@type="hidden") and not(contains(@name, "search"))'
    #         ' and not(contains(@placeholder, "search"))]'
    #     )
    #     search_pattern = re.compile(r'\b(search|query|find|keyword)\b', re.IGNORECASE)
    #     sensitive_inputs = [
    #         inp for inp in inputs
    #         if not search_pattern.search(inp.get('name', '') + inp.get('placeholder', ''))
    #     ]

    #     ## a login form will have at least 1 input box
    #     if len(sensitive_inputs) > 0:
    #         return True
    #     return False

    '''Phishpedia'''

    # @profile
    def test_orig_phishpedia(self, url, screenshot_path, html_path):
        # 0 for benign, 1 for phish, default is benign
        phish_category = 0
        pred_target = None
        matched_domain = None
        siamese_conf = None
        plotvis = None
        logo_match_time = 0
        print("Entering phishpedia")

        ####################### Step1: Logo detector ##############################################
        start_time = time.time()
        pred_boxes = pred_rcnn(im=screenshot_path, predictor=self.ELE_MODEL)
        logo_recog_time = time.time() - start_time

        if pred_boxes is not None:
            pred_boxes = pred_boxes.detach().cpu().numpy()
        plotvis = vis(screenshot_path, pred_boxes)

        # If no element is reported
        if pred_boxes is None or len(pred_boxes) == 0:
            print('No logo is detected')
            return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time

        ######################## Step2: Siamese (Logo matcher) ########################################
        start_time = time.time()
        pred_target, matched_domain, matched_coord, siamese_conf = check_domain_brand_inconsistency(
            logo_boxes=pred_boxes,
            domain_map_path=self.DOMAIN_MAP_PATH,
            model=self.SIAMESE_MODEL,
            logo_feat_list=self.LOGO_FEATS,
            file_name_list=self.LOGO_FILES,
            url=url,
            shot_path=screenshot_path,
            similarity_threshold=self.SIAMESE_THRE,
            topk=1)
        logo_match_time = time.time() - start_time

        if pred_target is None:
            print('Did not match to any brand, report as benign')
            return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time

        ######################## Step3: Simple input box check ###############
        # has_input_box = self.simple_input_box_regex(html_path=html_path)
        # if not has_input_box:
        # print('No input box')
        # return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time
        # else:
        print('Match to Target: {} with confidence {:.4f}'.format(pred_target, siamese_conf))
        phish_category = 1
        # Visualize, add annotations
        cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                    (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time


if __name__ == '__main__':

    '''update domain map'''
    # with open('./lib/phishpedia/models/domain_map.pkl', "rb") as handle:
    #     domain_map = pickle.load(handle)
    #
    # domain_map['weibo'] = ['sina', 'weibo']
    #
    # with open('./lib/phishpedia/models/domain_map.pkl', "wb") as handle:
    #     pickle.dump(domain_map, handle)
    # exit()

    '''run'''
    today = datetime.now().strftime('%Y%m%d')

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str)
    parser.add_argument("--output_txt", default=f'{today}_results.txt', help="Output txt path")
    args = parser.parse_args()

    request_dir = args.folder
    phishpedia_cls = PhishpediaWrapper()
    result_txt = args.output_txt

    os.makedirs(request_dir, exist_ok=True)

    for folder in tqdm(os.listdir(request_dir)):
        html_path = os.path.join(request_dir, folder, "html.txt")
        screenshot_path = os.path.join(request_dir, folder, "shot.png")
        info_path = os.path.join(request_dir, folder, 'info.txt')

        if not os.path.exists(screenshot_path):
            continue
        if not os.path.exists(html_path):
            html_path = os.path.join(request_dir, folder, "index.html")

        with open(info_path, 'r') as file:
            url = file.read()
        
        if os.path.exists(result_txt):
            with open(result_txt, 'r', encoding='ISO-8859-1') as file:
                if url in file.read():
                    continue

        _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|xml|sql|mdb|apk|bat|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|wks|wps|wpd)$"
        if re.search(_forbidden_suffixes, url, re.IGNORECASE):
            continue

        phish_category, pred_target, matched_domain, \
            plotvis, siamese_conf, pred_boxes, \
            logo_recog_time, logo_match_time = phishpedia_cls.test_orig_phishpedia(url, screenshot_path, html_path)

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                result_file_write(f, folder, url, phish_category, pred_target, matched_domain, siamese_conf,
                                  logo_recog_time, logo_match_time)
        except UnicodeError:
            with open(result_txt, "a+", encoding='utf-8') as f:
                result_file_write(f, folder, url, phish_category, pred_target, matched_domain, siamese_conf,
                                  logo_recog_time, logo_match_time)
        if phish_category:
            os.makedirs(os.path.join(request_dir, folder), exist_ok=True)
            cv2.imwrite(os.path.join(request_dir, folder, "predict.png"), plotvis)
    
