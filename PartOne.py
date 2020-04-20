import numpy as np
import sys
import os
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
#from detection import detections, calmAPAndPlot
from detection import *

conf_path = './config/conf.yaml'
with open(conf_path, 'r', encoding='utf-8') as f:
    data = f.read()
cfg = yaml.load(data)

gtFloder = 'data/groundtruths'
detFolder = 'data/detections'
save_path = 'data/results'

results, classes = detections(cfg, gtFloder, detFolder, save_path)
plot_save_result(cfg, results, classes, save_path)
