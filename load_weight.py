#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary
import os
from nets.yolo import YoloBody

from thop import profile



# if __name__ == "__main__":
#     device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # input = torch.randn(1, 3, 640, 640).to(device)
#     model       = YoloBody(80, 'm').to(device)
#     print(model.backbone)
#
#
#
#     if args.weights != "":
#         assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
#         weights_dict = torch.load(args.weights, map_location=device)["model"]
#         # 删除有关分类类别的权重
#         for k in list(weights_dict.keys()):
#             if "head" in k:
#                 del weights_dict[k]
#         print(model.load_state_dict(weights_dict, strict=False))
#     #
#     # summary(m, input_size=(3, 448, 448))


# 查看model其中的参数（非体系）
# model = YoloBody(num_classes=80, phi="l")
# for param_tensor in model.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
#     print(param_tensor,'\t',model.state_dict()[param_tensor].size())

# 查看.pth其中的参数(非体系)
# path = r"G:\学习资料\目标检测\yolox_l.pth"
# weight = torch.load(path)
# for name in weight.keys():
#     para = weight[name]
#     print(name,"   ", para.size())

model = YoloBody(num_classes=80, phi="l")
# 查看模型及参数
# print(model.backbone.backbone.state_dict())

# 查看具体某一层的详细信息，注意其与参数是有变化的
# print(model.backbone.backbone.patch_embed)
# print(type(model.backbone.backbone.patch_embed.proj.bias)) # nn.parameters
# print(type(model.backbone.backbone.patch_embed.proj.bias.data)) # tensor
# print(type(model.backbone.backbone.patch_embed.proj.bias.grad)) # 还没计算就是None

model_path = ""

if model_path != '':
    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#
    assert os.path.exists(model_path), "weights file: '{}' not exist.".format(model_path)
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # 删除有关分类类别的权重
    for k in list(model_dict.keys()):
        if "head" in k:
            del model_dict[k]
    print(model.load_state_dict(model_dict, strict=False))


