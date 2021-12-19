import itertools
import os
import os.path as osp
import time
from torchinfo import summary
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process

from tracker import matching

from tracker.basetrack import BaseTrack, TrackState

arch = 'dla_34'
ltrb = True
reid_dim = 128
head_conv = 256

num_classes = 1
heads = {'hm': num_classes,
        'wh': 2 if not ltrb else 4,
        'id': reid_dim}
MODEL_PATH = '/content/FairMOT/models/fairmot_dla34.pth'
model = create_model(arch, heads, head_conv)
start_epoch = 0
checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
print('loaded {}, epoch {}'.format(MODEL_PATH, checkpoint['epoch']))
state_dict_ = checkpoint['state_dict']
state_dict = {}
# convert data_parallal to model
for k in state_dict_:
  if k.startswith('module') and not k.startswith('module_list'):
    state_dict[k[7:]] = state_dict_[k]
  else:
    state_dict[k] = state_dict_[k]
model_state_dict = model.state_dict()

# check loaded parameters and created model parameters
msg = 'If you see this, your model does not fully load the ' + \
      'pre-trained weight. Please make sure ' + \
      'you have correctly specified --arch xxx ' + \
      'or set the correct --num_classes for your own dataset.'
counter = 0
for k in state_dict:
  counter += 1
  if k in model_state_dict:
    if state_dict[k].shape != model_state_dict[k].shape:
      print('Skip loading parameter {}, required shape{}, '\
            'loaded shape{}. {}'.format(
        k, model_state_dict[k].shape, state_dict[k].shape, msg))
      state_dict[k] = model_state_dict[k]
  else:
    print('Drop parameter {}.'.format(k) + msg)
for k in model_state_dict:
  if not (k in state_dict):
    print('No param {}.'.format(k) + msg)
    state_dict[k] = model_state_dict[k]
model.load_state_dict(state_dict, strict=False)
'''for param in model.parameters():
  param.requires_grad = False'''
batch_size = 32
print(summary(model, input_size=(batch_size, 3, 1920, 1080)))
