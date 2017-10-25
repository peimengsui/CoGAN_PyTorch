import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
  def __init__(self):
    super(BaseDataset, self).__init__()

  def name(self):
    return 'BaseDataset'

  def initialize(self, opt):
    pass


def get_transform():
  transform_list = []
  # TODO: Weird PIL nuance,  may change in newer version of pytorch so care.
  # osize = [opt.loadSizeW, opt.loadSizeH]
  # fsize = [opt.fineSizeH, opt.fineSizeW]
  # if opt.resize_or_crop == 'resize_and_crop':
  #   transform_list.append(transforms.Scale(osize, Image.BICUBIC))
  #   transform_list.append(transforms.RandomCrop(fsize))
  # elif opt.resize_or_crop == 'crop':
  #   transform_list.append(transforms.RandomCrop(fsize))
  # elif opt.resize_or_crop == 'scale_width':
  #   transform_list.append(transforms.Lambda(
  #       lambda img: __scale_width(img, opt.fineSize)))
  # elif opt.resize_or_crop == 'scale_width_and_crop':
  #   transform_list.append(transforms.Lambda(
  #       lambda img: __scale_width(img, opt.loadSize)))
  #   transform_list.append(transforms.RandomCrop(fsize))

  # if opt.isTrain and not opt.no_flip:
  #   transform_list.append(transforms.RandomHorizontalFlip())

  transform_list += [transforms.ToTensor()]
                     #transforms.Normalize((0.5, 0.5, 0.5),
                     #                     (0.5, 0.5, 0.5))]
  return transforms.Compose(transform_list)


def __scale_width(img, target_width):
  ow, oh = img.size
  if (ow == target_width):
    return img
  w = target_width
  h = int(target_width * oh / ow)
  return img.resize((w, h), Image.BICUBIC)
