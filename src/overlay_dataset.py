import os.path
import random
from pdb import set_trace as st

import numpy as np
import PIL
from PIL import Image
from skimage import img_as_ubyte, img_as_float
from skimage.io import imread

import torchvision.transforms as transforms
from base_dataset import BaseDataset, get_transform
from image_folder import make_dataset
#from DataUtils.add_noise_to_image import alpha_blend
from natsort import natsorted


class OverlayDataset(BaseDataset):
  def alpha_blend(self, noise_img, orig_img, alpha):
    return alpha * noise_img + (1 - alpha) * orig_img

  def initialize(self, config):
    # self.opt = opt
    # self.root = opt.dataroot
    # self.dir_orig = os.path.join(opt.dataroot, opt.phase, 'original')
    # self.dir_noise = os.path.join(opt.dataroot, opt.phase, 'texture')
    self.root = '/home/srivas/noise_mnist'
    # self.dir_orig = os.path.join(opt.dataroot, opt.phase, 'original')
    # self.dir_noise = os.path.join(opt.dataroot, opt.phase, 'texture')
    self.dir_orig = os.path.join(self.root, 'train', 'original')
    self.dir_noise = os.path.join(self.root, 'train', 'texture')

    self.orig_paths = make_dataset(self.dir_orig)
    self.noise_paths = make_dataset(self.dir_noise)

    self.orig_paths = natsorted(self.orig_paths)
    self.noise_paths = natsorted(self.noise_paths)
    self.orig_permute_paths = np.arange(len(self.orig_paths))
    #if opt.phase == 'train':
    self.orig_permute_paths = np.random.permutation(len(self.orig_paths))
    self.orig_size = len(self.orig_paths)
    self.transform = get_transform()
    self.alpha = config.alpha
    self.imagemode = 'L'

  def __getitem__(self, index):
    orig_path = self.orig_paths[index % self.orig_size]
    index_orig = index % self.orig_size
    orig_pair_path = self.orig_paths[self.orig_permute_paths[index_orig]]
    noise_path = self.noise_paths[self.orig_permute_paths[index_orig]]
    #print('(A, B) = (%d, %d)' % (index_A, index_B))
    # if self.opt.which_direction == 'BtoA':
    #   input_nc = self.opt.output_nc
    #   output_nc = self.opt.input_nc
    # else:
    # input_nc = self.opt.input_nc
    # output_nc = self.opt.output_nc
    input_nc = 1
    output_nc = 1

    # if input_nc == 1:
    #   orig_img = Image.open(orig_path).convert('RGB')
    # else:
    #   orig_img = Image.open(orig_path).convert('L')
    # if output_nc == 1:
    #   noise_img = Image.open(noise_path).convert('RGB')
    #   orig_pair_img = Image.open(orig_pair_path).convert('RGB')
    # else:
    orig_img = Image.open(orig_path).convert('L')
    noise_img = Image.open(noise_path).convert('L')
    orig_pair_img = Image.open(orig_pair_path).convert('L')
    A_img = self.transform(orig_img)
    blended_img = self.alpha_blend(np.array(img_as_float(noise_img)), np.array(img_as_float(orig_pair_img)), self.alpha)
    blended_img = img_as_ubyte(blended_img)
    B_img = self.transform(Image.fromarray(blended_img))

    # return {'A': A_img, 'B': B_img,
    #         'A_paths': orig_path, 'B_paths': orig_pair_path}
    return A_img, B_img

  def __len__(self):
    return self.orig_size

  def name(self):
    return 'OverlayDataset'
