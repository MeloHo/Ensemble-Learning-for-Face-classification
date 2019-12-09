from torch import nn
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, t, kernel_size,stride): ## Outchannel is for the third layer
        super(InvertedResidual, self).__init__()
        # t: expansion ratio
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.use_residual = stride == 1 and in_channels == out_channels
        layers = []

        padding = (kernel_size - 1) //2

        # First: 1 by 1 conv2d with ReLU6
        if (t != 1): # Because this layer is for expansino, if expansion is 1, there is no need for this layer
            conv1 = nn.Conv2d(in_channels, t*in_channels, kernel_size = 1, stride = 1, padding = 0)
            bn1 = nn.BatchNorm2d(t*in_channels)
            relu1 = nn.ReLU6(inplace = True)
            layers.append(conv1)
            layers.append(bn1)
            layers.append(relu1)
        # Second: Depthwise layer with ReLU6
        conv2 = nn.Conv2d(t*in_channels, t*in_channels, kernel_size = 3, stride = stride, padding = padding, groups = t*in_channels, bias = False)
        bn2 = nn.BatchNorm2d(t*in_channels)
        relu2 = nn.ReLU6(inplace = True)
        layers.append(conv2)
        layers.append(bn2)
        layers.append(relu2)
        # Third: Pointwise with Linear
        conv3 = nn.Conv2d(t*in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        bn3 = nn.BatchNorm2d(out_channels)
        layers.append(conv3)
        layers.append(bn3)

        self.conv = nn.Sequential(*layers)


        
    def forward(self, x):
        if self.use_residual:  # Using Residual
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV2(nn.Module):
    def __init__(self, num_classes = 2300):
        super(MobileNetV2, self).__init__()
        layers = []
        ## First Layer:
        # [n, 3, 32, 32]
        #input_conv = nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        input_conv = nn.Conv2d(3, 32, kernel_size = 1, stride = 1, padding = 0, bias = False)
        input_bn = nn.BatchNorm2d(32)
        input_relu = nn.ReLU6(inplace = True)
        layers.append(input_conv)
        layers.append(input_bn)
        layers.append(input_relu)

        # [n, 32, 16, 16]
        # BottleNeck1
        #layers.append(InvertedResidual(32, 16, t=1, kernel_size=3, stride=1))
        layers.append(InvertedResidual(32, 16, t=1, kernel_size=1, stride=1))
        #layers.append(nn.Dropout(p=0.2))



        # [n, 16, 16, 16]
        expansion = 6
        # BottleNeck2
        layers.append(InvertedResidual(16, 24, t=expansion, kernel_size=1, stride=1))
        #layers.append(InvertedResidual(24, 24, t=expansion, kernel_size=1, stride=1))
        #layers.append(InvertedResidual(16, 24, t=expansion, kernel_size=3, stride=2))
        #layers.append(InvertedResidual(24, 24, t=expansion, kernel_size=3, stride=1))
        #layers.append(InvertedResidual(24, 24, t=6, kernel_size=3, stride=1))
        #layers.append(nn.Dropout(p=0.2))


        # [n, 24, 8, 8]
        # BottleNeck3
        layers.append(InvertedResidual(24, 32, t=expansion, kernel_size=3, stride=1))
        layers.append(InvertedResidual(32, 32, t=expansion, kernel_size=3, stride=1))
        #layers.append(InvertedResidual(32, 32, t=expansion, kernel_size=3, stride=1))
        #layers.append(nn.Dropout(p=0.1))


        # [n, 32, 4, 4]
        # BottleNeck4
        layers.append(InvertedResidual(32, 64, t=expansion, kernel_size=3, stride=2))
        layers.append(InvertedResidual(64, 64, t=expansion, kernel_size=3, stride=1))
        #layers.append(InvertedResidual(64, 64, t=expansion, kernel_size=3, stride=1))
        #layers.append(InvertedResidual(64, 64, t=expansion, kernel_size=3, stride=1))

        # [n, 64, 4, 4]
        # BottleNeck5
        layers.append(InvertedResidual(64, 96, t=expansion, kernel_size=3, stride=2))
        #layers.append(InvertedResidual(96, 96, t=expansion, kernel_size=3, stride=1))
        #layers.append(InvertedResidual(96, 96, t=expansion, kernel_size=3, stride=1))

        # [n, 96, 4, 4]
        # BottleNeck6
        #layers.append(InvertedResidual(96, 160, t=expansion, kernel_size=3, stride=2))
        #layers.append(InvertedResidual(160, 160, t=expansion, kernel_size=3, stride=1))
        #layers.append(InvertedResidual(160, 160, t=expansion, kernel_size=3, stride=1))

        # [n, 160, 4, 4]
        # BottleNeck7
        #layers.append(InvertedResidual(160, 320, t=expansion, kernel_size=3, stride=1))






        output_conv = nn.Conv2d(96, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        output_bn = nn.BatchNorm2d(512)
        output_relu = nn.ReLU6(inplace = True)
        layers.append(output_conv)
        layers.append(output_bn)
        layers.append(output_relu)



        self.last_channel = 512
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


        self.feature = nn.Sequential(*layers)

        ### THIS PART IS FOR CENTER LOSS
        self.linear_closs = nn.Linear(512, 64, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        x = self.feature(x)
        x = x.mean([2, 3])
        # Center Loss
        #x_c = self.linear_closs(x)
        #x_c = self.relu_closs(x_c)
        ##
       # x = self.classifier(x)

        #return x_c, x
        return x
