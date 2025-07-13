import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset, train_transform
import os
from torchvision import transforms
import timm
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, dynamic_ncols=True, ascii=True):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')


test_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = ImageDataset(root_dir='/data1/zhangjiansong/py_project/Hosptial_other_project/Fujian_2/Doc_YongjianChen/LUSP-CLC-2024/val/', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
model = timm.create_model('resnet50', pretrained=False, num_classes=14)
model.load_state_dict(torch.load('./LUSP2024-14C-best_model.pth', map_location='cpu'))
model.cuda()
test_model(model, test_loader)
