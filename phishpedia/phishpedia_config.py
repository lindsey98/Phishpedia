# Global configuration
from .src.siamese import *
from .src.detectron2_pedia.inference import *

# element recognition model -- logo only
cfg_path = './src/detectron2_pedia/configs/faster_rcnn.yaml'
weights_path = './src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'
ele_model = config_rcnn(cfg_path, weights_path, conf_threshold=0.05)

# siamese model
print('Load protected logo list')
pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
                                                weights_path='./src/siamese_pedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='./src/siamese_pedia/expand_targetlist/')
print('Finish loading protected logo list')
print(logo_feat_list.shape)

siamese_ts = 0.83 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = './src/siamese_pedia/domain_map.pkl'

