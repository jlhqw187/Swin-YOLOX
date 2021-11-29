#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.yolo import YoloBody

from torchvision.models import resnet18
from thop import profile



if __name__ == "__main__":
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(1, 3, 448, 448).to(device)
    model       = YoloBody(80, 'l').to(device)
    flops, params = profile(model, inputs=(input,))
    print("Flops:",2 * flops / 1e9,"G  ", "params", params/1e6, "M")


    #
    # summary(model, input_size=(3, 448, 448))
