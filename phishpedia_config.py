# Global configuration
from src.siamese import *
from src.detectron2_pedia.inference import *
from src.util.chrome import vt_scan

# element recognition model -- logo only
cfg_path = './src/detectron2_pedia/configs/faster_rcnn.yaml'
weights_path = './src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'
ele_model = config_rcnn(cfg_path, weights_path, conf_threshold=0.05)

# siamese model
print('Load protected logo list')
pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
                                                weights_path='./src/phishpedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='./src/phishpedia/expand_targetlist/')
print('Finish loading protected logo list')

siamese_ts = 0.87 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = './src/phishpedia/domain_map.pkl'

