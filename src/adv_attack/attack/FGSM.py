import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import copy



def fgsm(model, method, image, label, criterion, max_iter=100, epsilon=0.05, clip_min=-1.0, clip_max=1.0):
    '''
    https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    FGSM attack
    :param model: subject model
    :param method: fgsm|stepll
    :param image: input image
    :param label: original class
    :param criterion: loss function to use
    :param max_iter: maximum iteration allowed
    :param epsilon: perturbation strength
    :param clip_min:  minimum/maximum value a pixel can take
    :param clip_max:
    :return: perturbed images
    '''

    # initialize perturbed image
    pert_image = copy.deepcopy(image)
    x = Variable(pert_image, requires_grad=True)

    output = model(x)
    pred = output.max(1, keepdim=True)[1]
    iter_ct = 0

    # loop until attack is successful
    while pred == label:
        if method == 'fgsm':
            loss = criterion(output, label)  # loss for ground-truth class
        else:
            ll = output.min(1, keepdim=True)[1][0]
            loss = criterion(output, ll)  # Loss for least-likely class

        # Back propogation
        zero_gradients(x)
        model.zero_grad()
        loss.backward()

        # Collect the sign of the data gradient
        sign_data_grad = torch.sign(x.grad.data.detach())

        # Create the perturbed image by adjusting each pixel of the input image
        if method == 'fgsm':
            x.data = x.data + epsilon * sign_data_grad
        else:
            x.data = x.data - epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range

        x.data = torch.clamp(x.data, clip_min, clip_max)
        output = model(x)
        pred = output.max(1, keepdim=True)[1]

        iter_ct += 1
        if iter_ct >= max_iter:
            break

    return x.data
