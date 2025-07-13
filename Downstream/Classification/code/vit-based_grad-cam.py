import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from grad_cam import GradCAM, show_cam_on_image,center_crop_img
import timm
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


output_dir = "/SZU_DATA/us-vfm/downstream_task/clc/code/breast-3c-grad-cam/"
# model_type = 'dinov2'
# model_type = 'MAE'
model_type = 'USFM'




Pretrained = True
def main():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)
    # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    Pretrained_path = '/SZU_DATA/us-vfm/downstream_task/clc/code/USFM-busi-3c-best_model.pth'
    if Pretrained and os.path.isfile(Pretrained_path) and model_type != 'USFM':
        print(f"=> loading checkpoint '{Pretrained_path}'")
        checkpoint = torch.load(Pretrained_path, map_location="cpu")
        state_dict = checkpoint.get('model', checkpoint)
        new_state_dict = {}

        # for k in list(state_dict.keys()):
        #     if k.startswith('model.base_encoder'):
        #         new_key = k.replace('model.base_encoder.', '')
        #         new_state_dict[new_key] = state_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        print('Loaded pretrained(encoder) ----->', Pretrained_path)
        print("Current model state_dict keys:")
        for k in model.state_dict().keys():
            print(k)

        # 打印预训练模型的层名称
        print("Loaded checkpoint state_dict keys:")
        for k in state_dict.keys():
            print(k)
    elif  Pretrained and os.path.isfile(Pretrained_path) and model_type == 'USFM':
        print(f"=> Loading USFM checkpoint '{Pretrained_path}'")
        checkpoint = torch.load(Pretrained_path, map_location="cpu")

        # 提取 state_dict，如果 checkpoint 没有 `state_dict`，直接使用 checkpoint 本身
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}

        # 遍历权重键，并处理前缀
        for k, v in state_dict.items():
            if k.startswith('module.base_encoder.'):
                new_key = k.replace('module.base_encoder.', '')  # 去掉不匹配的前缀
            elif k.startswith('base_encoder.'):
                new_key = k.replace('base_encoder.', '')
            else:
                new_key = k  # 保留原始键

            new_state_dict[new_key] = v

        # 尝试加载权重
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"=> Successfully loaded pretrained encoder from '{Pretrained_path}'")
        except RuntimeError as e:
            print(f"=> Error loading pretrained encoder: {e}")
            print("=> Retrying with strict=False")
            model.load_state_dict(new_state_dict, strict=False)
    elif Pretrained and os.path.isfile(Pretrained_path) and model_type == 'dinov2':
        import re
        import torch.nn.init as init

        print(f"=> Loading checkpoint '{Pretrained}'")
        checkpoint = torch.load(Pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('teacher.backbone.blocks.', 'blocks.')
            new_key = re.sub(r'\.\d+\.(?=\d)', r'.', new_key)
            new_key = new_key.replace('student.backbone.', '')
            # 检查权重是否包含 NaN 值
            if torch.isnan(v).any():
                print(f"Warning: Found NaN in weights for {new_key}. Reinitializing with Gaussian.")
                # 用高斯分布重新初始化，保持与原权重形状一致
                v = torch.empty_like(v)
                init.normal_(v, mean=0, std=0.02)  # 可以根据需求调整均值和标准差

            new_state_dict[new_key] = v

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained encoder from '{Pretrained}'")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print('dinov2')

    else:
        print('There is no pretrained for encoder')
    # model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    model.to(device)
    model.eval()

    target_layers = [model.blocks[-1].norm1]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = "/SZU_DATA/us-vfm/downstream_task/clc/data/Fetal_planes_8C/malignant (33).png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 2  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    # plt.imshow(visualization)
    # plt.show()
    output_path = os.path.join(output_dir, "fgsm_usfm_gradcam_result.png")
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM 结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
