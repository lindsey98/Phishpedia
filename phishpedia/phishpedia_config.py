# Global configuration
from phishpedia.src.siamese import *
from phishpedia.src.detectron2_pedia.inference import *
import phishpedia
import subprocess
from typing import Union
import yaml


def load_config(cfg_path: Union[str, None]):

    if cfg_path is None:
        with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(cfg_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

    ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH'].replace('/', os.sep)
    ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH'].replace('/', os.sep)
    ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
    ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)

    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

    print('Load protected logo list')
    targetlist_zip_path = configs['SIAMESE_MODEL']['TARGETLIST_PATH']
    targetlist_dir = os.path.dirname(targetlist_zip_path)
    zip_file_name = os.path.basename(targetlist_zip_path)
    targetlist_folder = zip_file_name.split('.zip')[0]
    full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)

    if targetlist_zip_path.endswith('.zip') and not os.path.isdir(full_targetlist_folder_dir.replace('/', os.sep)):
        os.makedirs(full_targetlist_folder_dir, exist_ok=True)
        subprocess.run(f'unzip -o "{targetlist_zip_path}" -d "{full_targetlist_folder_dir}"', shell=True)

    SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config(
                                                num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
                                                weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'].replace('/', os.sep),
                                                targetlist_path=full_targetlist_folder_dir.replace('/', os.sep))
    print('Finish loading protected logo list')

    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH'].replace('/', os.sep)

    return ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH




