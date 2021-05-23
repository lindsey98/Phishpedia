import torch
import torch.nn as nn
import torch.optim as optim


def cw(model, device, image, label, target_cls, c=1, kappa=0, max_iter=1000, learning_rate=0.05, verbose=1) :
    '''
    Implementation of C&W L2 targeted attack, Modified from https://github.com/Harry24k/CW-pytorch
    :param model: subject model
    :param device: cuda/cpu
    :param image: input image
    :param label: original class
    :param target_cls: adv class
    :param c: trade-off parameter to balance two optimization goals
    :param kappa: margin in confidence difference
    :param max_iter: maximimum iterations in optimization
    :param learning_rate: learning rate of optimizer
    :param verbose: print intermediate results or not
    :return: perturbed image
    '''
    
    # Get loss2
    def f(x) :
        
        output = model(x)
        one_hot_label = torch.eye(len(output[0]))[label].to(device)
        one_hot_target = torch.eye(len(output[0]))[target_cls].to(device)

        # confidence for the original predicted class and target class
        i, j = torch.masked_select(output, one_hot_label.bool()), torch.masked_select(output, one_hot_target.bool())
        
        # optimize for making the other class most likely 
        return torch.clamp(i-j, min=-kappa)

    
    # initialize w : the noise
    w = torch.zeros_like(image, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate) # an optimizer specifically for w

    for step in range(max_iter) :
        # w is the noise added to the original image, restricted to be [-1, 1]
        a = image + torch.tanh(w) 

        loss1 = nn.MSELoss(reduction='sum')(a, image) # ||x-x'||2
        loss2 = torch.sum(c*f(a)) # c*{f(label) - f(target_cls)}
        
        cost = loss1 + loss2 
        
        # Backprop: jointly optimize the loss
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # New prediction
        with torch.no_grad():
            pred_new = model(a)
            
        # Stop when ... 
        # successfully flip the label
        if torch.argmax(pred_new, dim=1).item() != label: 
            break

        if verbose > 0:
            print('- Learning Progress : %2.2f %% ' %((step+1)/max_iter*100), end='\r')
            
    # w is the noise added to the original image, restricted to be [-1, 1]
    attack_images = image + torch.tanh(w) 
    
    return attack_images


