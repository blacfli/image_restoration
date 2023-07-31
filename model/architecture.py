import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class OriginalModel(nn.Module):
    def __init__(self):
        super(OriginalModel, self).__init__()

        self.fc = nn.Sequential(nn.Linear(60, 256), nn.SiLU(),
                                nn.Linear(256, 512), nn.SiLU(),
                                nn.Linear(512, 256), nn.SiLU(),
                                nn.Linear(256, 60), nn.SiLU(),
                                nn.Linear(60, 4), nn.SiLU())
    
    def forward(self, x):
        out = self.fc(x)
        return out
    

class ConvModel1(nn.Module):
    def __init__(self):
        super(ConvModel1, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 2), nn.PReLU(),
                                     nn.Conv2d(32, 64, 2), nn.PReLU(),
                                     nn.AvgPool2d(2),
                                     nn.Conv2d(64, 128, 3), nn.PReLU())
        
        self.fc = self.fc = nn.Sequential(nn.Linear(128, 512),
                                nn.SiLU(),
                                nn.Linear(512, 256),
                                nn.SiLU(),
                                nn.Linear(256, 4),
                                )
    
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    

class ConvModel2(nn.Module):
    def __init__(self):
        super(ConvModel2, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 64, 3), nn.PReLU(),
                                     nn.AvgPool2d(2, stride=(1, 1)),
                                     nn.Conv2d(64, 128, 1), nn.PReLU(),
                                     nn.AvgPool2d(2, stride=(1, 1)),
                                     nn.Conv2d(128, 256, 3), nn.PReLU()
                                     )
        
        self.fc = self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.SiLU(),
                                nn.Linear(512, 256),
                                nn.SiLU(),
                                nn.Linear(256, 128),
                                nn.SiLU(),
                                nn.Linear(128, 4),
                                )
    
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    


class VGGSiameseNetwork(nn.Module):
    def __init__(self):
        super(VGGSiameseNetwork, self).__init__()

        # Load the ResNet18 model without pre-trained weights
        # resnet = torchvision.models.resnet18(weights=False)
        # print(resnet)
        self.vgg19_bn = torchvision.models.vgg19_bn(pretrained = True)
        self.vgg19_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        ct = 0
        for child in self.vgg19_bn.children():
            ct += 1
            if ct < 2:
                for param in child.parameters():
                    param.requires_grad = False
        # self.vgg19_bn = nn.Sequential(*(list(self.vgg19_bn.children())[:-1]))
        
        # print(list(self.vgg19_bn.children())[:-1])

        # Remove the last fully connected layer
        # self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Add new fully connected layers
        # self.classifier1 = nn.Sequential(
        #     nn.Linear(25088, 4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(4096, 4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(4096, 1000, bias=True),
        # )

    def forward_once(self, x):
        # Convert grayscale images to RGB images
        x = torch.cat([x, x, x], dim=1)
        output = self.vgg19_bn(x)
        # x = torch.cat([x, x, x], dim=1)
        # output = self.classifier1(output)
        # x = self.resnet(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
    
class VGGPretrainedSiameseNetwork(nn.Module):
    def __init__(self):
        super(VGGPretrainedSiameseNetwork, self).__init__()

        # self.vgg19_bn = torchvision.models.vgg19_bn(pretrained = True)
        # self.vgg19_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.vgg19_bn.classifier[-1] = nn.Linear(4096, 256, bias=True)

        # for name,child in self.vgg19_bn.named_children():
        #     if isinstance(child,nn.ReLU) or isinstance(child,nn.SELU):
        #         self.vgg19_bn._modules['relu'] = nn.SiLU()

        self.resnet = torchvision.models.resnet101(pretrained = True)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        for param in self.resnet.parameters():
            param.requires_grad = False
        # ct = 0
        # for child in self.vgg19_bn.children():
        #     ct += 1
        #     if ct < 2:
        #         for param in child.parameters():
        #             param.requires_grad = False

        self.fc = nn.Sequential(nn.Linear(self.fc_in_features, 512),
                                nn.SiLU(),
                                nn.Linear(512, 256),
                                nn.SiLU(),
                                nn.Linear(256, 256),
                                )

    def forward_once(self, x):
        # Convert grayscale images to RGB images
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
    


class CustomSiameseNetwork1(nn.Module):
    def __init__(self):
        super(CustomSiameseNetwork1, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(18432, 512),
                                nn.SiLU(),
                                nn.Linear(512, 256),
                                nn.SiLU(),
                                nn.Linear(256, 2),
                                )

    def forward_once(self, x):
        # Convert grayscale images to RGB images
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
    
class CustomSiameseNetwork2(nn.Module):
    def __init__(self):
        super(CustomSiameseNetwork2, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 128, 10), nn.SiLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(128, 256, 7), nn.SiLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(256, 512, 4), nn.SiLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(512, 1024, 4), nn.SiLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(1024, 512, 3), nn.SiLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(512, 256, 3), nn.SiLU(),
                                     nn.MaxPool2d(2)
                                    #  nn.Conv2d(256, 512, 1), nn.SiLU()
                                     )

        self.fc = nn.Sequential(nn.Linear(256, 512),
                                nn.SiLU(),
                                nn.Linear(512, 256),
                                nn.SiLU(),
                                nn.Linear(256, 256),
                                )

    def forward_once(self, x):
        # Convert grayscale images to RGB images
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
    
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(1, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.flatten = nn.Flatten()

    def forward_once(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        out = self.flatten(x)

        return out
  
    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


class PretrainedSiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(PretrainedSiameseNetwork, self).__init__()
        # get resnet model

        self.vgg19_bn = torchvision.models.vgg19_bn(pretrained = True)
        self.vgg19_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.vgg19_bn.classifier[-1] = nn.Linear(4096, 1, bias=True)

        # ct = 0
        # for child in self.vgg19_bn.children():
        #     ct += 1
        #     if ct < 2:
        #         for param in child.parameters():
        #             param.requires_grad = False

        self.vgg19_bn = nn.Sequential(*(list(self.vgg19_bn.children())[:-1]))
        # print(self.vgg19_bn)
        # for param in self.vgg19_bn.named_parameters():
        #     print(param)

        # self.resnet = torchvision.models.resnet101(pretrained = True)

        # # over-write the first conv layer to be able to read MNIST images
        # # as resnet18 reads (3,x,x) where 3 is RGB channels
        # # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.fc_in_features = self.resnet.fc.in_features
        
        # # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        # self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_in_features * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        # )

        self.fc = nn.Sequential(
            nn.Linear(25088, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        # self.resnet.apply(self.init_weights)
        # self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.vgg19_bn(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        # output = torch.cat((output1, output2), 1)
        output = output1 * output2

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        # output = self.sigmoid(output)
        
        return output