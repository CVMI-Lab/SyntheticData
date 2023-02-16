import os

import numpy as np

import torch


from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
# from ..src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier


# load_path = '/opt/tiger/filter_transfer/src/wise-ft/results/extest.r1/zeroshotEurosat.pt'
# load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/syn_init_tfda_55.68.pt'
# load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/syn_ref16.1_iters50_71.72.pt'
# load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/real_init_16.1_tfda_86.85.pt'
# load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/mix_16.1_iters50_88.21.pt'
# load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/mix_16.1_iters20_88.86.pt'
# load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/mix_16.1_v4_20k_87.86.pt'
load_path = '/opt/tiger/filter_transfer/pretrained-models/clip_few_shot/eurosat/mix_4.1_iter50_81.72.pt'
image_classifier = ImageClassifier.load(load_path)

# import ipdb
# ipdb.set_trace(context=20)

head = image_classifier.classification_head
weights = head.weight.detach().numpy()

torch.save(weights, '/opt/tiger/filter_transfer/src/wise-ft/cache/Eurosat/mix_4.1_iter50_81.72_weights.pt')
a=0
