
from phishpedia.phishpedia_main import *
import time
import datetime
import sys
from datetime import datetime, timedelta, time
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':

    ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(None)

    while True: # comment if you want to process it once
        # date = '2021-12-22'
        date = datetime.today().strftime('%Y-%m-%d')
        print('Today is:', date)
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', "--folder",
                            default='E:\\screenshots_rf\\{}'.format(date),
                            help='Input folder path to parse')
        parser.add_argument('-r', "--results", default=date + '_pedia.txt',
                            help='Input results file name')
        parser.add_argument('-t', '--repeat', action='store_true')
        args = parser.parse_args()
        print(args)
        runit(args.folder, args.results, ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
        print('Process finish')

        if not args.repeat:
            break
