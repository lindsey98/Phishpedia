import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import copy

def compute_jacobian(model, num_classes, inputs, output):
    '''
    'Helper function: compute jacobian matrix of confidence score vector w.r.t. input
    :param model: subject model
    :param num_classes:
    :param inputs:
    :param output:
    :return:
    '''

    jacobian = torch.zeros(num_classes, *inputs.size()).cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        model.zero_grad()
        output[0, i].backward(retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)



def saliency_map(jacobian, search_space, target_index):
    '''
    Helper function: compute saliency map and select the maximum index
    :param jacobian: jocabian matrix
    :param search_space: ignore some regions
    :param target_index: take one column of jacobian
    :return:
    '''
    
    jacobian = jacobian.squeeze(0)
    alpha = jacobian[target_index].sum(0).sum(0)
    beta = jacobian.sum(0).sum(0) - alpha

    # filter by the sign of alpha and beta
    mask1 = torch.ge(alpha, 0.0)
    mask2 = torch.le(beta, 0.0)
    mask = torch.mul(torch.mul(mask1, mask2), search_space)
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())

    # get the maximum index
    row_idx, col_idx = (saliency_map == torch.max(saliency_map)).nonzero()[0]
    return row_idx, col_idx


def jsma(model, num_classes, image, target, max_iter=100, clip_min=-1.0, clip_max=1.0):
    '''
    https://github.com/ast0414/adversarial-example/blob/master/craft.py
    Saliency map attack

    :param model: subject model
    :param num_classes: number of classes in classification
    :param image: input image
    :param target: original class
    :param max_iter: maximum iteration allowed
    :param clip_min: clip image into legal range
    :param clip_max:
    :return:
    '''

    # Make a clone since we will alter the values
    pert_image = copy.deepcopy(image)
    x = Variable(pert_image, requires_grad=True)

    output = model(x)
    label = output.max(1, keepdim=True)[1]

    count = 0
    
    # if attack is successful or reach the maximum number of iterations
    while label != target:

        # Skip the pixels that have been attacked before
        search_space = (x.data[0].sum(0) > clip_min*x.data.shape[1]) & (x.data[0].sum(0) < clip_max*x.data.shape[1])

        # Calculate Jacobian
        jacobian = compute_jacobian(model, num_classes, x, output)

        # get the highest saliency map's index
        row_idx, col_idx = saliency_map(jacobian, search_space, target)

        # increase to its maximum value
        x.data[0, :, row_idx, col_idx] = clip_max

        # recompute prediction
        output = model(x)
        label = output.max(1, keepdim=True)[1]

        count += 1
        if count >= max_iter:
            break

    return x.data


