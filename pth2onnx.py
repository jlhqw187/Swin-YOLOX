import os
import torch; print('Torch Version: {}'.format(torch.__version__))
import torch.onnx

from nets.yolo import YoloBody

model = YoloBody(num_classes=80, phi="l")

weights_path = r'G:\学习资料\目标检测'
weights_file_name = 'yolox_l.pth'
weights_file_path = os.path.join(weights_path, weights_file_name)

print('Loading weights file from:', weights_file_path)
state_dict = torch.load(weights_file_path)

model.load_state_dict(state_dict)

sample_batch_size, channel, height, width = 2, 3, 448, 448
dummy_input = torch.randn(sample_batch_size, channel, height, width)
output = model(dummy_input)

artifacts_path = r'G:\学习资料\目标检测'
onnx_file_name = weights_file_name.replace('pth', 'onnx')
onnx_file_path = os.path.join(artifacts_path, onnx_file_name)

print('Saving ONNX file to:', onnx_file_path)
torch.onnx.export(model, dummy_input, onnx_file_path)