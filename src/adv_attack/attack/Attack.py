from .FGSM import fgsm
from .JSMA import jsma
from .DeepFool import deepfool
from .CWL2 import cw
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class adversarial_attack():
    '''
    Perform adversarial attack
    '''
    def __init__(self, method, model, dataloader, device, num_classes=10, save_data=False):
        '''
        :param method: Which attack method to use
        :param model: subject model to attack
        :param dataloader: dataloader
        :param device: cuda/cpu
        :param num_classes: number of classes for classification model
        :param save_data: save data or not
        '''
        self.method = method
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.save_data = save_data
        
    def batch_attack(self):
        '''
        Run attack on a batch of data
        
        '''
        # Accuracy counter
        correct = 0
        total = 0
        adv_examples = []
        ct_save = 0
#         adv_cat = torch.tensor([])

        # Loop over all examples in test set
        for ct, (data, label) in tqdm(enumerate(self.dataloader)):
            data = data.to(self.device, dtype=torch.float) 
            label = label.to(self.device, dtype=torch.long)
            
            # Forward pass the data through the model
            output = self.model(data)
            self.model.zero_grad()
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            if init_pred.item() != label.item():  # initially was incorrect --> no need to generate adversary
                total += 1
                print(ct)
                continue

            # Call Attack
            if self.method in ['fgsm', 'stepll']:
                criterion = nn.CrossEntropyLoss()
                perturbed_data = fgsm(self.model, self.method, data, label, criterion)
                
            elif self.method == 'jsma':
                # randomly select a target class
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                print(target_class)
                perturbed_data = jsma(self.model, self.num_classes, data, target_class)
                
            elif self.method == 'deepfool':
                f_image = output.detach().cpu().numpy().flatten()
                I = (np.array(f_image)).flatten().argsort()[::-1]
                perturbed_data = deepfool(self.model, self.num_classes, data, label, I)
                
            elif self.method == 'cw':
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                print(target_class)
                perturbed_data = cw(self.model, self.device, data, label, target_class)
                
            else:
                print('Attack method is not supported， please choose your attack from [fgsm|stepll|jsma|deepfool|cw]')
                
                
            # Re-classify the perturbed image
            self.model.zero_grad()
            self.model.eval()
            with torch.no_grad():
                output = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == init_pred.item():
                correct += 1  # still correct
            else:# save successful attack
                print(final_pred)
                print(init_pred)
                if self.save_data:
                    os.makedirs('./data/normal_{}'.format(self.method), exist_ok=True)
                    os.makedirs('./data/adversarial_{}'.format(self.method), exist_ok=True)
                    # Save the original instance
                    torch.save((data.detach().cpu(), init_pred.detach().cpu()),
                               './data/normal_{}/{}.pt'.format(self.method, ct_save))
                    # Save the adversarial example
                    torch.save((perturbed_data.detach().cpu(), final_pred.detach().cpu()),
                               './data/adversarial_{}/{}.pt'.format(self.method, ct_save))

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            total += 1
            print(ct)
            print("Test Accuracy = {}".format(correct/float(total)))

        # Calculate final accuracy
        final_acc = correct / float(len(self.dataloader))
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
            
