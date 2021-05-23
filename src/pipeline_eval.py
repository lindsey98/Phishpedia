import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use all devices

from tqdm import tqdm
import time
from src.siamese import *
from src.detectron2_pedia.inference import *
import argparse
import errno


def phishpedia_eval(data_dir, mode, siamese_ts, write_txt):
    '''
    Run phishpedia evaluation
    :param data_dir: data folder dir
    :param mode: phish|benign
    :param siamese_ts: siamese threshold
    :param write_txt: txt path to write results
    :return:
    '''
    with open(write_txt, 'w') as f:
        f.write('folder\t')
        f.write('true_brand\t')
        f.write('phish_category\t')
        f.write('pred_brand\t')
        f.write('runtime_element_recognition\t')
        f.write('runtime_siamese\n')

    for folder in tqdm(os.listdir(data_dir)):

        phish_category = 0  # 0 for benign, 1 for phish
        pred_target = None  # predicted target, default is None

        img_path = os.path.join(data_dir, folder, 'shot.png')
        html_path = os.path.join(data_dir, folder, 'html.txt')
        if mode == 'phish':
            url = eval(open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read())
            url = url['url'] if isinstance(url, dict) else url
        else:
            try:
                url = open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read()
            except:
                url = 'https://www' + folder

        # Element recognition module
        start_time = time.time()
        pred_boxes, _, _, _ = pred_rcnn(im=img_path, predictor=ele_model)
        pred_boxes = pred_boxes.detach().cpu().numpy()
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0  # Report as benign

        # If at least one element is reported
        else:
            # Phishpedia module
            start_time = time.time()
            pred_target, _, _ = phishpedia_classifier_logo(logo_boxes=pred_boxes, domain_map_path=domain_map_path,
                                                           model=pedia_model,
                                                           logo_feat_list=logo_feat_list,
                                                           file_name_list=file_name_list,
                                                           url=url,
                                                           shot_path=img_path,
                                                           ts=siamese_ts)

            siamese_time = time.time() - start_time

            # Phishpedia reports target
            if pred_target is not None:
                phish_category = 1  # Report as suspicious

            # Phishpedia does not report target
            else:  # Report as benign
                phish_category = 0

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder + '\t')
            f.write(brand_converter(folder.split('+')[0]) + '\t')  # true brand
            f.write(str(phish_category) + '\t')  # phish/benign/suspicious
            f.write(brand_converter(pred_target) + '\t') if pred_target is not None else f.write('\t')  # phishing target
            # element recognition time
            f.write(str(ele_recog_time) + '\t')
            # siamese time
            f.write(str(siamese_time) + '\n') if 'siamese_time' in locals() else f.write('\n')

            # delete time variables
        try:
            del ele_recog_time
            del siamese_time
        except:
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--mode", choices=['phish', 'benign', 'discovery'], required=True,
                        help="Evaluate phishing or benign or discovery")
    parser.add_argument("--write-txt", required=True, help="Where to save results")
    parser.add_argument("--data-dir", required=True, help="Data Dir")
    parser.add_argument("--ts", required=True, help="Siamese threshold")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)
    mode = args.mode
    siamese_ts = float(args.ts)
    write_txt = args.write_txt

    # element recognition model -- logo only
    cfg_path = 'src/detectron2_pedia/configs/faster_rcnn.yaml'
    weights_path = 'src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'
    ele_model = config_rcnn(cfg_path, weights_path, conf_threshold=0.05)

    # Siamese
    pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
                                                                    weights_path='src/phishpedia/resnetv2_rgb_new.pth.tar',
                                                                    targetlist_path='src/phishpedia/expand_targetlist/')
    print('Number of protected logos = {}'.format(str(len(logo_feat_list))))

    # Domain map path
    domain_map_path = 'src/phishpedia/domain_map.pkl'

    # PhishPedia
    phishpedia_eval(data_dir, mode, siamese_ts, write_txt)