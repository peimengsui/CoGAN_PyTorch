from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

imsize = 28

loader = transforms.Compose([transforms.ToTensor()])  # transform it into a torch tensor
root = '/home/srivas/noise_mnist'
dir_orig = os.path.join(root, 'train', 'original')
dir_noise = os.path.join(root, 'train', 'texture')

def alpha_blend(noise_img, orig_img, alpha):
    return alpha * noise_img + (1 - alpha) * orig_img

def image_loader(orig_image_name, noise_image_name):
	
    orig_img = Image.open(orig_image_name).convert('L')
    noise_img = Image.open(noise_image_name).convert('L')
    blended_img = alpha_blend(np.array(img_as_float(noise_img)), np.array(img_as_float(orig_pair_img)), alpha)
    blended_img = img_as_ubyte(blended_img)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    #image = image.unsqueeze(0)
    return image


style_img = image_loader("images/picasso.jpg").type(dtype)
content_img = image_loader("images/dancing.jpg").type(dtype)

class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() 
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

network = torch.load('trainer_alpha0.5.pt')
network.cuda()

def get_generated_image(network, noised_img):
	network = copy.deepcopy(network)
    noise = Variable(torch.randn(1, 100)).cuda()
    fake_images_a, fake_images_b = network.gen_update(noise)
    content_loss = ContentLoss(fake_images_b)
    model = nn.Sequential()
    model.cuda()
    model.add_module("content_los_" + str(i), content_loss)
    #model = ContentLoss(noised_img)
    return fake_images_b, model, noise

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_optimization(network, noised_img, num_steps=300):
    """Run the style transfer."""
    fake_images_b, model, input_img = get_generated_image(network, noised_img)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            content_loss = model(input_param)
            content_score = 0
            content_score += content_loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Content Loss: {:4f}'.format(content_score.data[0]))
                print()

            return content_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data

######################################################################
# Finally, run the algorithm

output = run_optimization(network, noised_img)