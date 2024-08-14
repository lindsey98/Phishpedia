# Global configuration
import yaml
from logo_matching import cache_reference_list, load_model_weights
import os
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)


def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))

def load_config(reload_targetlist=False):

    with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the configuration and update paths
    for section, settings in configs.items():
        for key, value in settings.items():
            if 'PATH' in key and isinstance(value, str):  # Check if the key indicates a path
                absolute_path = get_absolute_path(value)
                configs[section][key] = absolute_path

    ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
    # Load a model
    ELE_MODEL = YOLO(
        ELE_WEIGHTS_PATH
    )  # load a pretrained model (recommended for training)

    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

    # logging.info('Load protected logo list')
    # targetlist_zip_path = configs['SIAMESE_MODEL']['TARGETLIST_PATH']
    # targetlist_dir = os.path.dirname(targetlist_zip_path)
    # zip_file_name = os.path.basename(targetlist_zip_path)
    # targetlist_folder = zip_file_name.split('.zip')[0]
    # full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)
    #
    # if reload_targetlist or targetlist_zip_path.endswith('.zip') and not os.path.isdir(full_targetlist_folder_dir):
    #     os.makedirs(full_targetlist_folder_dir, exist_ok=True)
    #     subprocess.run(f'unzip -o "{targetlist_zip_path}" -d "{full_targetlist_folder_dir}"', shell=True)

    SIAMESE_MODEL = load_model_weights(weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'])

    # if reload_targetlist or (not os.path.exists(os.path.join(os.path.dirname(__file__), 'LOGO_FEATS.npy'))):
    #     LOGO_FEATS, LOGO_FILES = cache_reference_list(model=SIAMESE_MODEL,
    #                                                   targetlist_path=full_targetlist_folder_dir)
    #     logging.info('Finish loading protected logo list')
    #     np.save(os.path.join(os.path.dirname(__file__),'LOGO_FEATS.npy'), LOGO_FEATS)
    #     np.save(os.path.join(os.path.dirname(__file__),'LOGO_FILES.npy'), LOGO_FILES)
    #
    # else:
    #     LOGO_FEATS, LOGO_FILES = np.load(os.path.join(os.path.dirname(__file__),'LOGO_FEATS.npy')), \
    #                              np.load(os.path.join(os.path.dirname(__file__),'LOGO_FILES.npy'))

    LOGO_FEATS, LOGO_FILES = np.load(os.path.join(os.path.dirname(__file__), 'LOGO_FEATS.npy')), np.load(os.path.join(os.path.dirname(__file__),'LOGO_FILES.npy'))
    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH