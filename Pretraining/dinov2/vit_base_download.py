import timm
import torch
import os

# 确保权重保存目录存在
os.makedirs('./vit_b_img21k_weights', exist_ok=True)

# 加载预训练模型
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=21843)

# 保存模型权重
weight_path = './vit_b_img21k_weights/vit_base_16_imagenet21k.pth'

# 包装状态字典以匹配加载期望的结构
torch.save(model.state_dict(), weight_path)

print(f"Model weights saved to {weight_path}")

#
# print(f"Model weights saved to {weight_path}")
# Load the PyTorch model weights file
model_weights_path = '/SZU_DATA/us-vfm/upstream_task/dinov2/vit_b_img21k_weights/vit_base_16_imagenet21k.pth'
checkpoint = torch.load(model_weights_path, map_location='cpu')

# Print the keys in the checkpoint file
print("Keys in the checkpoint file:", checkpoint.keys())
