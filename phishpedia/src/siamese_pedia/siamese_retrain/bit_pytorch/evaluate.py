
import torch
import numpy as np
import phishpedia.src.siamese_pedia.siamese_retrain.bit_pytorch.models as models
from phishpedia.src.siamese_pedia.siamese_retrain.bit_pytorch.dataloader import GetLoader
import torch.nn.functional as F
import os
import cv2
import matplotlib.pyplot as plt
import torchvision as tv

def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)

            # Compute output, measure accuracy
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y).sum().item()
            total += len(logits)
            print("GT:", y)
            print("Pred:", preds)
            print(correct, total)

    return float(correct/total)

def heatmap_vis(x, savedir):
    # Visualize grid array
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot("331")
    ax.set_title("X", fontsize=15)
    ax.imshow(x[0].numpy() if not isinstance(x[0], np.ndarray) else x[0], cmap='gray')

    ax = plt.subplot("332")
    ax.set_title("Y", fontsize=15)
    ax.imshow(x[1].numpy() if not isinstance(x[1], np.ndarray) else x[1], cmap='gray')

    ax = plt.subplot("333")
    ax.set_title("W", fontsize=15)
    ax.imshow(x[2].numpy() if not isinstance(x[2], np.ndarray) else x[2], cmap='gray')

    ax = plt.subplot("334")
    ax.set_title("H", fontsize=15)
    ax.imshow(x[3].numpy() if not isinstance(x[3], np.ndarray) else x[3], cmap='gray')

    ax = plt.subplot("335")
    ax.set_title("C1(logo)", fontsize=15)
    ax.imshow(x[4].numpy() if not isinstance(x[4], np.ndarray) else x[4], cmap='gray')

    ax = plt.subplot("336")
    ax.set_title("C2(input)", fontsize=15)
    ax.imshow(x[5].numpy() if not isinstance(x[5], np.ndarray) else x[5], cmap='gray')

    ax = plt.subplot("337")
    ax.set_title("C3(button)", fontsize=15)
    ax.imshow(x[6].numpy() if not isinstance(x[6], np.ndarray) else x[6], cmap='gray')

    ax = plt.subplot("338")
    ax.set_title("C4(label)", fontsize=15)
    ax.imshow(x[7].numpy() if not isinstance(x[7], np.ndarray) else x[7], cmap='gray')

    ax = plt.subplot("339")
    ax.set_title("C5(block)", fontsize=15)
    ax.imshow(x[8].numpy() if not isinstance(x[8], np.ndarray) else x[8], cmap='gray')

    plt.savefig(savedir)
    plt.close()


def evaluate_special(model, dataset):

    bad_example_dir = './datasets/bad'
    if len(os.listdir(bad_example_dir)) > 0:
        for file in os.listdir(bad_example_dir):
            os.unlink(os.path.join(bad_example_dir, file))
    os.makedirs(bad_example_dir, exist_ok=True)

    model.eval()
    for j in range(len(dataset)):
        filepath = list(set(dataset.paths))[j]
        img_coords = np.asarray(dataset.preprocess_coordinates)[np.asarray(dataset.paths) == filepath] # box coordinates
        img_classes = np.asarray(dataset.img_classes)[np.asarray(dataset.paths) == filepath] # box types
        # print(img_coords)
        x, y = dataset.__getitem__(j)
        with torch.no_grad():
            pred = model(x[None, ...].type(torch.float).to(device)).detach().cpu()
            pred_score = F.softmax(pred, dim=-1)
            pred_cls = pred_score.argmax(-1).item()

        # Visualize incorrect examples
        if pred_cls != y:
            print('File {} has predicted class {:d} with predicted score {:.2f}'.format(filepath, pred_cls, pred_score[:, pred_cls].item()))
            img = cv2.imread(os.path.join('./datasets/val_imgs', filepath+'.png'))
            print(img.shape)
            for coord in img_coords:
                x1, y1, x2, y2 = coord
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)

            cv2.imwrite(os.path.join(bad_example_dir, filepath + "_{}_{}_".format('credential' if pred_cls == 0 else 'noncredential', str(pred_score[:, pred_cls].item())) + '.png'), img)

            heatmap_vis(x, os.path.join(bad_example_dir, filepath + '_heatmap_xywh_c5_.png'))


def cls_avg_heatmaps(cls, dataset):
    ct = 0
    channel_heatmaps = np.zeros((9, 10, 10))  # xywhc1..c5 channel

    for j in range(len(dataset)):
        x, y = dataset.__getitem__(j) # jth sample
        if y == cls:
            ct += 1
            for k in range(9): # kth channel
                channel_heatmaps[k] = channel_heatmaps[k] + x[k].numpy()

    avg_channel_heatmaps = channel_heatmaps / ct
    heatmap_vis(avg_channel_heatmaps, savedir='./output/cls{}_avg_heatmap.png'.format(str(cls)))
    return avg_channel_heatmaps



if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    val_tx = tv.transforms.Compose([
            tv.transforms.Resize((95, 95)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_set = GetLoader(data_root='./src/siamese_pedia/expand_targetlist',
                          data_list='./src/siamese_pedia/siamese_retrain/test_targets.txt',
                          label_dict='./src/siamese_pedia/siamese_retrain/target_dict.json',
                          transform=val_tx)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), drop_last=False, shuffle=False)
    
    model = models.KNOWN_MODELS['BiT-M-R50x1'](head_size=len(val_set.classes), zero_head=True)
    checkpoint = torch.load('./src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth', map_location="cpu")["model"]
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)

    acc = evaluate(model, val_loader)
    print(acc)
