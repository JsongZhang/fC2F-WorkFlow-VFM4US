import os
import torch
import cv2
import numpy as np
import timm
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from grad_cam import GradCAM, show_cam_on_image, center_crop_img  # 引入你提供的 GradCAM 实现

# 配置路径
model_path = "/SZU_DATA/us-vfm/downstream_task/clc/code/fgsm_resnet18-busi-3c-best_model.pth"
image_path = "/SZU_DATA/us-vfm/downstream_task/clc/data/Fetal_planes_8C/malignant (33).png"
output_dir = "/SZU_DATA/us-vfm/downstream_task/clc/code/breast-3c-grad-cam"


os.makedirs(output_dir, exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # 加载 ResNet18 模型
    model = timm.create_model('resnet18', pretrained=False, num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layers = [model.layer4[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/SZU_DATA/us-vfm/downstream_task/clc/data/Fetal_planes_8C/malignant (33).png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 3  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    # plt.imshow(visualization)
    # plt.show()

    output_path = os.path.join(output_dir, "adv3_us4fm-gradcam_result.png")
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM 结果已保存至: {output_path}")


if __name__ == '__main__':
    main()


