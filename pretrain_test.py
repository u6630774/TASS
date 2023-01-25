from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import attacks

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from torch.autograd import Variable
import IAA

model = network.modeling.__dict__["deeplabv3plus_resnet50"](num_classes=21, output_stride=16)
model.load_state_dict( torch.load( "pretrained/best_deeplabv3plus_resnet50_voc_os16.pth")['model_state'])