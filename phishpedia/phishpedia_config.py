# Global configuration
from phishpedia.src.siamese import *
from phishpedia.src.detectron2_pedia.inference import *
import subprocess

# element recognition model -- logo only
cfg_path = os.path.join(os.path.dirname(__file__), 'src/detectron2_pedia/configs/faster_rcnn.yaml')
weights_path = os.path.join(os.path.dirname(__file__), 'src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth')
ele_model = config_rcnn(cfg_path, weights_path, conf_threshold=0.05)

# siamese model
print('Load protected logo list')
subprocess.run(
    "unzip {}/src/siamese_pedia/expand_targetlist.zip -d {}/src/siamese_pedia/expand_targetlist/".format(os.path.dirname(__file__), os.path.dirname(__file__)),
    shell=True,
)
pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
                                                weights_path=os.path.join(os.path.dirname(__file__), 'src/siamese_pedia/resnetv2_rgb_new.pth.tar'),
                                                targetlist_path=os.path.join(os.path.dirname(__file__),'src/siamese_pedia/expand_targetlist/'))
print('Finish loading protected logo list')
print(logo_feat_list.shape)

siamese_ts = 0.83 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = os.path.join(os.path.dirname(__file__), 'src/siamese_pedia/domain_map.pkl')

