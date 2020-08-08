from __future__ import print_function
from numpy import pi, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .resnet_leaky import ResNet18

BIAS_VALUE = None


class BasicCNN(nn.Module):
    def __init__(self, noutputs=10):
        super(BasicCNN, self).__init__()
        self.name = "CNN"
        self.nfeatures = 500
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, self.nfeatures)
        self.fc2 = nn.Linear(self.nfeatures, noutputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)  # or 4*4 for 28x28 images
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# FCN described in MNIST experiments: https://arxiv.org/abs/1711.08856
# (CLP = Critical Learning Pds)
class CLPFCN(nn.Module):
    def __init__(self):
        super(CLPFCN, self).__init__()
        self.name = "CLPFCN"
        self.net = nn.Sequential(
            nn.Linear(1024, 2500),
            nn.BatchNorm1d(num_features=2500),
            nn.ReLU(),
            nn.Linear(2500, 2000),
            nn.BatchNorm1d(num_features=2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.BatchNorm1d(num_features=1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU()
        )
        self.fc = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.net(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class BasicFCN(nn.Module):
    def __init__(self):
        super(BasicFCN, self).__init__()
        self.name = "FCN"
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LinearFCN(nn.Module):
    def __init__(self):
        super(LinearFCN, self).__init__()
        self.name = "LinearFCN"
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return F.log_softmax(self.fc1(x), dim=1)


class AllConvNet(nn.Module):
    def __init__(self, input_size=32, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)

    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# TODO: USE DICT INSTEAD OF MANY IFs
def pick_model(**kwargs):
    model_name = kwargs["name"]
    input_size = kwargs.get("input_size", 32)
    num_classes = kwargs["num_outputs"]

    feature_extract = kwargs.get("feature_extract", False)
    negative_slope = kwargs.get("slope", 0.0)
    use_pretrained: bool = kwargs.get("use_pretrained", False)

    global BIAS_VALUE
    if "use_bias_reset" in kwargs:
        BIAS_VALUE = kwargs["use_bias_reset"]
    else:
        BIAS_VALUE = None
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "clp_mnist_fcn":
        model_ft = CLPFCN()
        # input_size = 1024

    elif model_name == "allconv":
        model_ft = AllConvNet()
        # input_size = 32

    elif model_name == "resnet":
        """ Resnet18
        """
        # input size of torchvision Resnet18: 224x224
        # model_ft = ResNet18(num_classes=num_classes)
        # input size for resnet_leaky: 32

        if use_pretrained:
            # !! we are using the torchvision Resnet18 to use a pretrained model !!
            # It is not the same model as our local resnet_leaky.py as the architecture is tuned for ImageNet
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

            # replace ReLUs with LeakyReLUs if requested
            if negative_slope > 0.0:
                replace_relu_with_leaky_relu(model_ft, negative_slope=negative_slope)
        else:
            # this model architecture of Resnet18 is adjusted for CIFAR10 and input images of size 32
            model_ft = ResNet18(input_size=input_size, num_classes=num_classes, negative_slope=negative_slope)

        if BIAS_VALUE is not None:
            model_ft.apply(reset_bias)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        # input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def reset_bias(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d or type(m) == torch.nn.BatchNorm2d:
        if m.bias is not None:
            m.bias.data.fill_(torch.max(m.bias.data) + BIAS_VALUE)


def replace_relu_with_leaky_relu(m: nn.Module, negative_slope: float = 0.0):
    layers_to_replace = []
    for layer_name, layer in m.named_children():
        if isinstance(layer, nn.ReLU):
            layers_to_replace.append(layer_name)
        else:
            replace_relu_with_leaky_relu(layer, negative_slope=negative_slope)

    for layer_name in layers_to_replace:
        setattr(m, layer_name, nn.LeakyReLU(negative_slope=negative_slope))
