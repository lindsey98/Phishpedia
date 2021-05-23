from detectron2.configs import get_cfg
import detectron2.data.transforms as T
import torch
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import cv2
import argparse

def load_cfg(cfg_path, weights_path):
    '''
    Load configs
    '''
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg_clone = cfg.clone()
    cfg_clone.MODEL.WEIGHTS = weights_path
    return cfg_clone



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help='Detectron2 config file', required=True)
    parser.add_argument("--weights", help='Detectron2 weights file', required=True)
    parser.add_argument("--dataset", help="Dataset name", required=True)
    args = parser.parse_args()
    
    # load config and dataset
    cfg_clone = load_cfg(args.config_file, args.weights)
    data_loader = build_detection_test_loader(
        cfg_clone, args.dataset
    )
    
    # test-time transformation
    aug = T.ResizeShortestEdge(
                [cfg_clone.INPUT.MIN_SIZE_TEST, cfg_clone.INPUT.MIN_SIZE_TEST],
                cfg_clone.INPUT.MAX_SIZE_TEST)
    
    # Runs on entire test dataset
    # For each image

    for i, batch in tqdm(enumerate(data_loader)):
        original_image = batch[0]["image"].permute(1, 2, 0).numpy()
        file_name = batch[0]["file_name"]
        basename = os.path.basename(file_name)
        image_id = batch[0]["image_id"]

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            model = build_model(cfg_clone)
            model.eval()
            images = model.preprocess_image(batch)
            features = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, features)

            pred_objectness_logits = []
            for x in features.values():
                pred_objectness_logits.append(model.proposal_generator.rpn_head.objectness_logits(x))
            pred_objectness = [F.sigmoid(x) for x in pred_objectness_logits]
    
        # Plot RPN regions
    #     print('File: ', file_name)
    #     check = images.tensor[0].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]
    #     check = cv2.UMat(check).get()
    #     proposal_bbox = proposals[0].proposal_boxes.tensor.detach().cpu().numpy()
    #     objectness = F.sigmoid(proposals[0].objectness_logits).detach().cpu().numpy()

    #     print('Number of proposal: ', len(proposal_bbox))
    #     for i, box in enumerate(proposal_bbox):
    #         cv2.rectangle(check, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), 
    #                       color=np.random.rand(3,)*255, thickness=3)
    #         cv2.putText(check, str(round(objectness[i], 2)), (int(box[0]), int(box[1]+10)), 
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Plot objectness map
        
        # resize objectness map to overlay with original image
        resize_shape = (images[0].permute(1, 2, 0).detach().cpu().shape[1], 
                        images[0].permute(1, 2, 0).detach().cpu().shape[0])
        
        # Plot objectness map
        fig, axs = plt.subplots(3, 2, figsize=(30, 30))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            if i == 0:
                ax.imshow(original_image[:, :, ::-1])
                ax.set_title('Input image', fontsize=30)
            else:
                p = cv2.UMat(pred_objectness[i-1][0][0].detach().cpu().numpy()).get() # get P2-P6
                ax.imshow(original_image[:, :, ::-1])
                ax.imshow(cv2.resize(p, resize_shape), cmap='coolwarm', alpha=0.7) # overlay resized objectness map
                ax.set_title('P{}'.format(str(i+1)), fontsize=30)

#         if file_name.split('/')[-1] in os.listdir('../datasets/login_test_correct/'):
#             plt.savefig('../datasets/login_test_correct_RPN/'+file_name.split('/')[-1]) 
#             plt.close(fig)
            
#         elif file_name.split('/')[-1] in os.listdir('../datasets/login_test_wrong/'):
#             plt.savefig('../datasets/login_test_wrong_RPN/'+file_name.split('/')[-1]) 
#             plt.close(fig)        
