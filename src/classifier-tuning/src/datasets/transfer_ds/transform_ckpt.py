from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def transform_ckpt_attacker2clean():
    attacker_path = '/opt/tiger/filter_transfer/pretrained-models/resnet-18-100k-distilled-ex20.2.e2.ckpt'
    attacher_dict = torch.load(attacker_path)
    attacher_dict = attacher_dict['model']
    teacher_state_dict = OrderedDict()
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in attacher_dict.items():
        if 'module.attacker.model' in k:
            teacher_state_dict[k.replace('module.attacker.model.', '')] = v
            # pass
        elif 'module.model' in k:
            # teacher_state_dict[k.replace('module.model.', '')] = v
            pass
        else:
            print('exception, key name is %s' % k)
            # teacher_state_dict[k] = v

    # torch.save(teacher_state_dict, 'transformed_'+attacker_path)
    torch.save(teacher_state_dict, attacker_path + '.clean')

def transform_ckpt_attacker2custom():

    # model -> model.model
    attacker_path = '/opt/tiger/filter_transfer/pretrained-models/RN50.clean.attacker'
    attacher_dict = torch.load(attacker_path)
    attacher_dict = attacher_dict['model']
    teacher_state_dict = OrderedDict()
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in attacher_dict.items():
        if 'module.attacker.model' in k:
            teacher_state_dict[k.replace('module.attacker.model', 'module.attacker.model.model')] = v
            # pass
        elif 'module.model' in k:
            teacher_state_dict[k.replace('module.model', 'module.model.model')] = v
            pass
        else:
            print('exception, key name is %s' % k)
            teacher_state_dict[k] = v
            # teacher_state_dict[k] = v


    # torch.save(teacher_state_dict, 'transformed_'+attacker_path)
    to_save = {}
    to_save['model'] = teacher_state_dict
    to_save['epoch'] = 0

    torch.save(to_save, attacker_path + '.custom')
    print('saving to %s' % attacker_path + '.custom')

    # torch.save(teacher_state_dict, attacker_path + '.custom')

def transform_ckpt_res18_2_res18_feat():
    from resnet import resnet50, resnet18_feat
    path = '../pretrained-models/transformed_resnet-50-l2-eps0.ckpt'


    student = resnet18_feat(pretrained=False)
    st = student.state_dict()
    # model -> model.model
    attacker_path = '/opt/tiger/filter_transfer/pretrained-models/res18_injected_from_res50_imgnet.ckpt'

    # import ipdb
    # ipdb.set_trace(context=20)

    attacher_dict = torch.load(attacker_path)
    st.update(attacher_dict)


    # torch.save(teacher_state_dict, 'transformed_'+attacker_path)
    to_save = {}
    to_save['model'] = st
    to_save['epoch'] = 0

    torch.save(to_save, attacker_path + '.res_feat')
    print('saving to %s' % attacker_path + '.res_feat')

    # torch.save(teacher_state_dict, attacker_path + '.custom')

def transform_ckpt_custom2attacker(exp_name):
    # model.model  -> model
    attacker_path = '../outdir/'+exp_name+'/checkpoint.pt.latest'
    attacher_dict = torch.load(attacker_path)
    attacher_dict = attacher_dict['model']
    teacher_state_dict = OrderedDict()
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in attacher_dict.items():
        if ('emb' in k) or ('l2norm' in k):
            print('exception: abandom, key name is %s' % k)
        elif 'module.attacker.model.model' in k:
            teacher_state_dict[k.replace('module.attacker.model.model', 'module.attacker.model')] = v
            # pass
        elif 'module.model.model' in k:
            teacher_state_dict[k.replace('module.model.model', 'module.model')] = v
            pass
        else:
            print('exception:same, key name is %s' % k)
            teacher_state_dict[k] = v
            # teacher_state_dict[k] = v

    # torch.save(teacher_state_dict, 'transformed_'+attacker_path)
    to_save = {}
    to_save['model'] = teacher_state_dict
    to_save['epoch'] = 0

    torch.save(to_save, attacker_path + '.attacker')
    print('saving to %s' % attacker_path + '.attacker')

    # torch.save(teacher_state_dict, attacker_path + '.custom')

def transform_ckpt_custom2seg():
    # model.model  ->
    attacker_path = '../pretrained-models/resnet-18-l2-eps0.ckpt.custom'
    attacher_dict = torch.load(attacker_path)
    attacher_dict = attacher_dict['model']
    teacher_state_dict = OrderedDict()
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in attacher_dict.items():
        if ('emb' in k) or ('l2norm' in k):
            print('exception: abandom, key name is %s' % k)
        elif 'module.attacker.model.model' in k:
            teacher_state_dict[k.replace('module.attacker.model.model.', '')] = v
            # pass
        elif 'module.model.model' in k:
            teacher_state_dict[k.replace('module.model.model.', '')] = v
            pass
        else:
            print('exception:same, key name is %s' % k)
            teacher_state_dict[k] = v

    torch.save(teacher_state_dict, attacker_path + '.seg')
    print('saving to %s' % attacker_path + '.seg')

    # torch.save(teacher_state_dict, attacker_path + '.custom')


def transform_ckpt_clean2attacker():
    from resnet import resnet50, resnet18_feat
    from robustness import datasets, defaults, model_utils
    from clip import RN50

    # path = '../outdir/ex44.2/checkpoint.pt.latest'
    path = '../pretrained-models/RN50.clean'
    ckpt = torch.load(path)

    # model = resnet18_feat(pretrained=False)
    model = RN50(pretrained=False)
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        # if 'attacker' in k:
        new_ckpt[k.replace('module.', '')] = v
    # import ipdb
    # ipdb.set_trace(context=20)
    model.load_state_dict(new_ckpt, strict=False)

    model, _ = model_utils.make_and_restore_model(arch=model, dataset=datasets.ImageNet(''), add_custom_forward=True)

    to_save = {}
    sd = model.state_dict()

    teacher_state_dict = OrderedDict()
    for k, v in sd.items():
        # if 'attacker' in k:
        teacher_state_dict['module.' + k.replace('model.model', 'model')] = v
        # elif 'module.model' in k:
        #     teacher_state_dict[k.replace('module.model.', '')] = v
        # else:
        #     print('exception, key name is %s' % k)
    to_save['model'] = teacher_state_dict
    to_save['epoch'] = 0

    torch.save(to_save, path + '.attacker')
    print('saving to %s' % path + '.attacker')

    q = 0

def transform_ckpt_res50_to_res18():

    # 'res18: '
    # (['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
    # 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
    # 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean',
    # 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
    # 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight',
    # 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer2.0.conv1.weight',
    # 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked',
    # 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var',
    # 'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias',
    # 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight',
    # 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight',
    # 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', '
    # layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var',
    # 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean',
    # 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight',
    # 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked',
    # 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var',
    # 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean',
    # 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias',
    # 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight',
    # 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight',
    # 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var',
    # 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean',
    # 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
    # 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'fc.weight', 'fc.bias'])

    # res50:
    # ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.num_batches_tracked', 'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias', 'layer1.0.downsample.1.running_mean', 'layer1.0.downsample.1.running_var', 'layer1.0.downsample.1.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.conv3.weight', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var', 'layer2.0.bn3.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv3.weight', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.2.conv3.weight', 'layer2.2.bn3.weight', 'layer2.2.bn3.bias', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked', 'layer2.3.conv3.weight', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv3.weight', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.conv3.weight', 'layer3.1.bn3.weight', 'layer3.1.bn3.bias', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.2.conv3.weight', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked', 'layer3.3.conv3.weight', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.num_batches_tracked', 'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.4.conv3.weight', 'layer3.4.bn3.weight', 'layer3.4.bn3.bias', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.4.bn3.num_batches_tracked', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer3.5.conv3.weight', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias', 'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var', 'layer3.5.bn3.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv3.weight', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.bn3.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var', 'layer4.1.bn3.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.num_batches_tracked', 'fc.weight', 'fc.bias']

    from resnet import resnet50, resnet18
    path = '../pretrained-models/transformed_resnet-50-l2-eps0.ckpt'

    ckpt = torch.load(path)

    student = resnet18(pretrained=False, deep_base=False)
    st = student.state_dict()
    # print(st.keys())
    # import ipdb
    # ipdb.set_trace(context=20)

    def get_feats(state_dict, key):
        results = []
        catalog = OrderedDict()
        for k, v in state_dict.items():
            if key in k and len(v.shape) == 4:
                if v.shape[3] == 1:
                    continue
                else:
                    n, c, _a, _b = v.shape
                    # results.append(v.view(n * c, 3 * 3))
                    # import ipdb
                    # ipdb.set_trace(context=20)
                    results.append(v)
                    catalog[k] = v.shape
        # return torch.cat(results)
        return results

    ckpt_s = OrderedDict()

    print("start neck...")
    ckpt_s['conv1.weight'] = ckpt['conv1.weight']
    ckpt_s['bn1.weight'] = ckpt['bn1.weight']
    ckpt_s['bn1.bias'] = ckpt['bn1.bias']
    ckpt_s['bn1.running_mean'] = ckpt['bn1.running_mean']
    ckpt_s['bn1.running_var'] = ckpt['bn1.running_var']
    ckpt_s['bn1.num_batches_tracked'] = ckpt['bn1.num_batches_tracked']

    print("start layer1...")
    layer1_convs = get_feats(ckpt, 'layer1')
    # import ipdb
    # ipdb.set_trace(context=20)
    layer1_avg = sum(layer1_convs) / len(layer1_convs)

    ckpt_s['layer1.0.conv1.weight'] = layer1_avg
    ckpt_s['layer1.0.conv2.weight'] = layer1_avg
    ckpt_s['layer1.1.conv1.weight'] = layer1_avg
    ckpt_s['layer1.1.conv2.weight'] = layer1_avg

    print("start layer2...")
    layer2_convs = get_feats(ckpt, 'layer2')
    layer2_avg = sum(layer2_convs) / len(layer2_convs)
    ckpt_s['layer2.0.conv1.weight'] = layer2_avg[:,:64,:,:]
    ckpt_s['layer2.0.conv2.weight'] = layer2_avg
    ckpt_s['layer2.1.conv1.weight'] = layer2_avg
    ckpt_s['layer2.1.conv2.weight'] = layer2_avg

    print("start layer3...")
    layer3_convs = get_feats(ckpt, 'layer3')
    layer3_avg = sum(layer3_convs) / len(layer3_convs)
    ckpt_s['layer3.0.conv1.weight'] = layer3_avg[:,:128,:,:]
    ckpt_s['layer3.0.conv2.weight'] = layer3_avg
    ckpt_s['layer3.1.conv1.weight'] = layer3_avg
    ckpt_s['layer3.1.conv2.weight'] = layer3_avg

    print("start layer4...")
    layer4_convs = get_feats(ckpt, 'layer4')
    layer4_avg = sum(layer4_convs) / len(layer4_convs)
    # import ipdb
    # ipdb.set_trace(context=20)
    ckpt_s['layer4.0.conv1.weight'] = layer4_avg[:,:256,:,:]
    ckpt_s['layer4.0.conv2.weight'] = layer4_avg
    ckpt_s['layer4.1.conv1.weight'] = layer4_avg
    ckpt_s['layer4.1.conv2.weight'] = layer4_avg

    st.update(ckpt_s)
    torch.save(st, '../pretrained-models/res18_injected_from_res50_imgnet.ckpt')
    print('saving to %s' % '../pretrained-models/res18_injected_from_res50_imgnet.ckpt')

    # import ipdb
    # ipdb.set_trace(context=20)
    # a=1




if __name__ == "__main__":
    import sys
    exp_name = sys.argv[1]
    print(sys.argv[1])
    # transform_ckpt_attacker2clean()
    # transform_ckpt_clean2attacker()
    # transform_ckpt_res50_to_res18()
    # transform_ckpt_attacker2custom()
    transform_ckpt_custom2attacker(exp_name)
    # transform_ckpt_custom2seg()
    # transform_ckpt_res18_2_res18_feat()