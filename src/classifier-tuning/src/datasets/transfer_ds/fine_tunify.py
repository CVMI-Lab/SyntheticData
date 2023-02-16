from torch import nn
from robustness.tools.custom_modules import SequentialWithArgs

def ft(model_name, model_ft, num_classes, additional_hidden=0):
    if model_name in ['clip_resnest50d','resnet50_feat_pca_pre_relu_multi_pool','resnet18_feat_pre_relu_multi_pool','resnet50_feat_pca_pre_relu_multi','resnet18_feat_pre_relu_multi','resnet50_feat_interpolate_multi','resnet18_multi','resnet50_feat_pre_relu_multi','resnet50_overhaul','resnet18_feat_pre_relu_regressor','resnet18_custom','resnet18_feat_pre_relu',"resnet50_feat_interpolate","resnet50_feat_pca","resnet50_feat_nmf","resnet50_feat_lda","resnet50_feat_mag","resnet18_feat","resnet152_feat","resnet50_feat","resnet","resnet20_as_gift","resnet50_clean", "resnet18", "resnet34","resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet']:
        num_ftrs = model_ft.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        if additional_hidden == 0:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )
        input_size = 224

    elif model_name == 'RN50':
        num_ftrs = 1024
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif "vgg" in model_name:
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name in ["mnasnet", "mobilenet"]:
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        pass
        # raise ValueError("Invalid model type, exiting...")

    return model_ft
