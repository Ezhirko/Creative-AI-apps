import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN18K(nn.Module):
  def __init__(self):
    super(SimpleCNN18K,self).__init__()
    self.conv1 = nn.Conv2d(1,8,3,stride=1,padding=1) # input_size = 28, output_size = 28, RF = 3,
    self.conv1_bn = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8,16,3,stride=1,padding=0)# input_size = 28, output_size = 26, RF = 5
    self.conv2_bn = nn.BatchNorm2d(16)
    self.drop1 = nn.Dropout2d(0.05)
    self.pool1 = nn.MaxPool2d(2,2)                    # input_size = 26, output_size = 13, RF = 6
    self.conv3 = nn.Conv2d(16,16,3,stride=1,padding=1)# input_size = 13, output_size = 13, RF = 10
    self.conv3_bn = nn.BatchNorm2d(16)
    self.drop2 = nn.Dropout2d(0.05)
    self.conv4 = nn.Conv2d(16,32,3,stride=1,padding=0)# input_size = 13, output_size = 11, RF = 14
    self.conv4_bn = nn.BatchNorm2d(32)
    self.drop3 = nn.Dropout2d(0.05)
    self.pool2 = nn.MaxPool2d(2,2)                      # input_size = 11, output_size = 5, RF = 16
    self.conv5 = nn.Conv2d(32,32,3,stride=1,padding=0)# input_size = 5, output_size = 3, RF = 24
    self.conv5_bn = nn.BatchNorm2d(32)
    self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # reduces the spatial dimensions to 1x1
    self.conv6 = nn.Conv2d(32,10,1)                   # input_size = 3, output_size = 1, RF = 40

  def forward(self,x):
    x = self.pool1(F.relu(self.drop1(self.conv2_bn(self.conv2(F.relu(self.conv1_bn(self.conv1(x))))))))
    x = self.pool2(F.relu(self.drop3(self.conv4_bn(self.conv4(F.relu(self.drop2(self.conv3_bn(self.conv3(x)))))))))
    x = self.pool3(F.relu(self.conv5_bn(self.conv5(x))))
    x = self.conv6(x)
    x = x.view(-1,10)
    return F.log_softmax(x)