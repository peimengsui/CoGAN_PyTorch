from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte, io
import torchvision.transforms as transforms
import torchvision.models as models

import copy

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

imsize = 28
alpha = 0.9
loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor
root = '/home/srivas/noise_mnist'
dir_orig = os.path.join(root, 'test', 'original')
dir_noise = os.path.join(root, 'test', 'texture')

def alpha_blend(noise_img, orig_img, alpha):
    return alpha * noise_img + (1 - alpha) * orig_img

def image_loader(orig_image_name, noise_image_name):
	
    orig_img = Image.open(orig_image_name).convert('L')
    noise_img = Image.open(noise_image_name).convert('L')
    blended_img = alpha_blend(np.array(img_as_float(noise_img)), np.array(img_as_float(orig_img)), alpha)
    blended_img = img_as_ubyte(blended_img)
    image = Variable(loader(blended_img.reshape(imsize,imsize,1)))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image



class ContentLoss(nn.Module):

    def __init__(self, network, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() 
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.criterion = nn.MSELoss()
	self.network = network

    def forward(self, input):
	self.loss = self.criterion(self.network.gen(input)[1], self.target.cuda())
        self.output = self.network.gen(input)[1]
        return self.output

    def backward(self, retain_graph=True):
        #self.loss.backward(retain_graph=retain_graph)
        self.loss.backward()
	return self.loss

network = torch.load('trainer_alpha{}.pt'.format(alpha))
network.cuda()

def get_model_loss(network, noised_img):
    #noise = Variable(torch.randn(1, 100)).cuda()
    #fake_images_a, fake_images_b = network.gen_update(noise)
    content_loss = ContentLoss(network, noised_img)
    model = nn.Sequential()
    model.add_module("content_loss", content_loss)
    model.cuda()
    return  model, content_loss

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_optimization(network, noised_img, num_steps=200):
    """Run the style transfer."""
    model, content_loss = get_model_loss(network, noised_img)
    input_img = Variable(torch.randn(1, 100)).cuda()
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            #input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            content_score = 0
            content_score += content_loss.backward()

            run[0] += 1
            if run[0] % 5 == 0:
                print("run {}:".format(run))
                print('Content Loss: {:4f}'.format(content_score.data[0]))
                print()

            return content_score

        optimizer.step(closure)

    # a last correction...
    #input_param.data.clamp_(0, 1)

    return input_param.data

######################################################################
# Finally, run the algorithm
def imsave(tensor, alpha, i):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(1, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    image.save('../denoised/'+str(alpha)+'/{}.png'.format(i))
for i in range(10000):
    noised_img = image_loader(os.path.join(root, dir_orig)+'/{}.png'.format(i), os.path.join(root, dir_noise)+'/{}.png'.format(i))
    output = run_optimization(network, noised_img)
    result = network.gen(Variable(output).cuda())[0].data.cpu().squeeze(0)
    unloader = transforms.ToPILImage()
    imsave(result, alpha, i)


