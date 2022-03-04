# Global configuration
from phishpedia.src.siamese import *
from phishpedia.src.detectron2_pedia.inference import *
import subprocess
from typing import Union
import yaml


def load_config(cfg_path: Union[str, None]):

    #################### '''Default''' ####################
    if cfg_path is None:
        with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        # element recognition model -- logo only
        ELE_CFG_PATH = os.path.join(os.path.dirname(__file__), configs['ELE_MODEL']['CFG_PATH'])
        ELE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), configs['ELE_MODEL']['WEIGHTS_PATH'])
        ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
        ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)

        # siamese model
        SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

        print('Load protected logo list')
        if configs['SIAMESE_MODEL']['TARGETLIST_PATH'].endswith('.zip') \
                and not os.path.isdir('{}/{}'.format(os.path.dirname(__file__),
                                                     configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0])
                                      ):
            subprocess.run(
                "unzip {}/{} -d {}/{}/".format(os.path.dirname(__file__),
                                               configs['SIAMESE_MODEL']['TARGETLIST_PATH'],
                                               os.path.dirname(__file__),
                                               configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0]
                                               ),
                shell=True,
            )

        SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config(
            num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
            weights_path=os.path.join(os.path.dirname(__file__), configs['SIAMESE_MODEL']['WEIGHTS_PATH']),
            targetlist_path=os.path.join(os.path.dirname(__file__),
                                         configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0]))
        print('Finish loading protected logo list')
        DOMAIN_MAP_PATH = os.path.join(os.path.dirname(__file__), configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH'])

    #################### '''Customized''' ####################
    else:
        with open(cfg_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH']
        ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
        ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
        ELE_MODEL = config_rcnn(ELE_CFG_PATH, ELE_WEIGHTS_PATH, conf_threshold=ELE_CONFIG_THRE)

        # siamese model
        SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

        print('Load protected logo list')
        if configs['SIAMESE_MODEL']['TARGETLIST_PATH'].endswith('.zip') \
                and not os.path.isdir('{}'.format(configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0])):
            subprocess.run(
                "unzip {} -d {}/".format(configs['SIAMESE_MODEL']['TARGETLIST_PATH'],
                                         configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0]),
                shell=True,
            )

        SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config(
            num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
            weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
            targetlist_path=configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0])
        print('Finish loading protected logo list')

        DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH




