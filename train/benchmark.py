import os

from tqdm import tqdm
import time
import argparse
import errno
from phishpedia import PhishpediaWrapper
import itertools

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--mode', type=str, default='phish', choices=['phish', 'benign'],
                        help='Mode of operation, can be phish or benign')
    parser.add_argument('--threshold', type=float, default=0.85, help='Threshold value for decision making')
    args = parser.parse_args()

    phishpedia_cls = PhishpediaWrapper()
    mode = args.mode
    threshold = args.threshold

    data_dir = f'./datasets/{mode}_sample_30k'
    result_txt = f'./train/benchmark30k_{mode}_{threshold}.txt'
    phishpedia_cls.SIAMESE_THRE = threshold  # reset threshold

    for folder in tqdm(os.listdir(data_dir)):

        screenshot_path = os.path.join(data_dir, folder, 'shot.png')
        html_path = os.path.join(data_dir, folder, 'html.txt')
        if mode == 'phish':
            url = eval(open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read())
            url = url['url'] if isinstance(url, dict) else url
        else:
            try:
                url = open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read()
            except:
                url = 'https://www.' + folder

        if os.path.exists(result_txt) and url in open(result_txt, encoding='ISO-8859-1').read():
            continue

        phish_category, pred_target, matched_domain, \
            plotvis, siamese_conf, pred_boxes, \
            logo_recog_time, logo_match_time = phishpedia_cls.test_orig_phishpedia(url, screenshot_path, html_path,
                                                                                   False)
        print(logo_recog_time + logo_match_time)

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(url + "\t")
                f.write(str(phish_category) + "\t")
                f.write(str(pred_target) + "\t")  # write top1 prediction only
                f.write(str(matched_domain) + "\t")
                f.write(str(siamese_conf) + "\t")
                f.write(str(round(logo_recog_time, 4)) + "\t")
                f.write(str(round(logo_match_time, 4)) + "\n")
        except UnicodeError:
            with open(result_txt, "a+", encoding='utf-8') as f:
                f.write(folder + "\t")
                f.write(url + "\t")
                f.write(str(phish_category) + "\t")
                f.write(str(pred_target) + "\t")  # write top1 prediction only
                f.write(str(matched_domain) + "\t")
                f.write(str(siamese_conf) + "\t")
                f.write(str(round(logo_recog_time, 4)) + "\t")
                f.write(str(round(logo_match_time, 4)) + "\n")

        if len(open(result_txt).readlines()) >= 500:
            break
        # break

    ### FIXME: Google has a lot of domain alias