import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import copy

def deepfool(model, num_classes, image, label, I, overshoot=0.02, max_iter=100, clip_min=-1.0, clip_max=1.0):
    '''
    https://github.com/LTS4/DeepFool/tree/master/Python
    DeepFool attack
    :param model: subject model
    :param num_classes: number of classes in classification
    :param image: input image
    :param label: original class
    :param I: current predicted class ranked by decending order
    :param overshoot: scale factor to increase perturbation a little bit
    :param max_iter: maximum iterations allowed
    :param clip_min: clip image into legal range
    :param clip_max: clip image into legal range
    :return: perturbed image
    '''

    pert_image = copy.deepcopy(image)
    w = np.zeros(image.shape)
    r_tot = np.zeros(image.shape)

    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    fs = model(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    # Stop until attack is successful or reach the maximum iterations
    while k_i == label:

        pert = np.inf
        zero_gradients(x)
        model.zero_grad()
        fs[0, I[0]].backward(retain_graph=True) # backpropogate the maximum confidence score
        grad_orig = x.grad.data.detach().cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)
            model.zero_grad()
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.detach().cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.detach().cpu().numpy()

            if np.linalg.norm(w_k.flatten()) == 0.0: # if w_k is all zero, no perturbation at all
                pert_k = 0.0 * abs(f_k)
            else:
                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        if np.linalg.norm(w) == 0.0:
            r_i = (pert+1e-4) * w
        else:
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        pert_image = torch.clamp(pert_image, clip_min, clip_max)

        x = Variable(pert_image, requires_grad=True)
        fs = model(x)
        k_i = np.argmax(fs.data.detach().cpu().numpy().flatten())

        loop_i += 1
        if loop_i >= max_iter:
            break

    return x.data

