# Global configuration
import os
import zipfile
import shutil
import numpy as np
import yaml
from logo_matching import cache_reference_list, load_model_weights
from logo_recog import config_rcnn


def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))


def extract_targetlist(zip_path, target_dir):
    '''
    Extract the protected-logo target list from its zip archive if it has not
    been extracted yet, flattening any redundant nested top-level folder.
    :param zip_path: path to expand_targetlist.zip
    :param target_dir: directory the archive should be extracted into
    '''
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return  # already extracted

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Target list not found. Expected an extracted folder at '{target_dir}' "
            f"or the archive '{zip_path}'. Run the setup script to download the models."
        )

    print(f'Extracting target list from {zip_path} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    # Flatten a redundant nested "<name>/<name>/..." directory if present.
    nested = os.path.join(target_dir, os.path.basename(target_dir))
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            shutil.move(os.path.join(nested, entry), os.path.join(target_dir, entry))
        os.rmdir(nested)


def load_config(reload_targetlist=False):
    with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the configuration and update paths
    for section, settings in configs.items():
        for key, value in settings.items():
            if 'PATH' in key and isinstance(value, str):  # Check if the key indicates a path
                absolute_path = get_absolute_path(value)
                configs[section][key] = absolute_path

    ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH']
    ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
    ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
    ELE_MODEL = config_rcnn(ELE_CFG_PATH,
                            ELE_WEIGHTS_PATH,
                            conf_threshold=ELE_CONFIG_THRE)

    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

    print('Load protected logo list')
    targetlist_zip_path = configs['SIAMESE_MODEL']['TARGETLIST_PATH']
    targetlist_dir = os.path.dirname(targetlist_zip_path)
    targetlist_folder = os.path.basename(targetlist_zip_path).split('.zip')[0]
    full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)

    # Auto-extract the target list if only the zip archive is present.
    extract_targetlist(targetlist_zip_path, full_targetlist_folder_dir)

    SIAMESE_MODEL = load_model_weights(num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
                                       weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'])

    LOGO_FEATS_NAME = 'LOGO_FEATS.npy'
    LOGO_FILES_NAME = 'LOGO_FILES.npy'

    if reload_targetlist or (not os.path.exists(os.path.join(os.path.dirname(__file__), LOGO_FEATS_NAME))):
        LOGO_FEATS, LOGO_FILES = cache_reference_list(model=SIAMESE_MODEL,
                                                      targetlist_path=full_targetlist_folder_dir)
        print('Finish loading protected logo list')
        # Store as arrays so the freshly-computed result matches the cached-load
        # path (downstream code relies on numpy array semantics, e.g. .shape).
        LOGO_FEATS, LOGO_FILES = np.array(LOGO_FEATS), np.array(LOGO_FILES)
        np.save(os.path.join(os.path.dirname(__file__), LOGO_FEATS_NAME), LOGO_FEATS)
        np.save(os.path.join(os.path.dirname(__file__), LOGO_FILES_NAME), LOGO_FILES)

    else:
        LOGO_FEATS, LOGO_FILES = np.load(os.path.join(os.path.dirname(__file__), LOGO_FEATS_NAME)), \
            np.load(os.path.join(os.path.dirname(__file__), LOGO_FILES_NAME))

    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH
