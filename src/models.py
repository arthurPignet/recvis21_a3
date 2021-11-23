import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CNNScattering(nn.Module):
    '''
        Simple CNN with 3x3 convs based on VGG used for the scattering input
    '''
    def __init__(self, in_channels):
        super(CNNScattering, self).__init__()
        self.in_channels = in_channels
        self.build()

    def build(self):
        cfg = [128, 128, 'M', 64, 64]
        layers = []
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                self.in_channels = v

        layers += [nn.AdaptiveAvgPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(-1,nclasses)

    def forward(self, x):
        x = self.bn(x.view(-1, self.K, 16, 16))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class Decoder(nn.Module):
    def __init__(self, latent_space_dim):
        super(Decoder, self).__init__()

        # fc block
        self.fc = nn.Sequential(nn.Linear(latent_space_dim, 4096),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, 25088),
                                nn.Dropout(p=0.5, inplace=False),
                                nn.ReLU(inplace=True)
                                )

        # first block
        self.first_block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, 3, 1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        # second block
        self.second_block = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # nn.ReflectionPad2d((1, 1, 1, 1)),
            # nn.Conv2d(256, 256, 3, 1, 0),
            # nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),
        )
        # third block
        self.third_block = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),
        )
        # fourth block
        self.fourth_block = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3, 1, 0),
        )

    def forward(self, input):
        out = self.fc(input)
        out = out.view(-1, 512, 7, 7)

        out = self.first_block(out)
        out = self.second_block(out)
        out = self.third_block(out)
        out = self.fourth_block(out)

        return out


class VGG13AE(nn.Module):
    def __init__(self, latent_space_dim, feature_extract=False, use_pretrained=True):
        super(VGG13AE, self).__init__()

        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        self.input_size = 224
        self.encoder = models.vgg13_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(self.encoder, feature_extract)
        num_ftrs = self.encoder.classifier[6].in_features
        self.encoder.classifier[6] = nn.Linear(num_ftrs, latent_space_dim)

        self.classifier = nn.Sequential(nn.Linear(latent_space_dim, nclasses))

        self.decoder = Decoder(latent_space_dim=latent_space_dim)

        self.mode_autoencoder = True

    def forward(self, x):
        x = self.encoder(x)
        if self.mode_autoencoder:
            x = self.decoder(x)
        else:
            x = self.classifier(x)
        return x

    def cuda(self):
        self.classifier.cuda()
        self.encoder.cuda()
        self.decoder.cuda()


def (model_name, num_classes=20, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    def set_parameter_requires_grad(model, feature_extracting):
      if feature_extracting:
          for param in model.parameters():
              param.requires_grad = False

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg13_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "regnet":
        """ regnet
        """
        model_ft = models.regnet_y_32gf(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size