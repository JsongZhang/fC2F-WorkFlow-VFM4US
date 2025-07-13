import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from dataset import train_dataset, val_dataset
from BUSI_dataset import busi_train_dataset, busi_val_dataset
import os
import random
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Pretrained = True
Pretrained_path = '/SZU_DATA/us-vfm/dinov2_output_us120k_us380kpretrain/checkpoint_124999.pth'
# Pretrained_path = '/data1/zhangjiansong/py_project/VFM/USFM-2024MIA/USFMpretrained.ckpt'


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id) #seed_num + work_id
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, dynamic_ncols=True, ascii=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return avg_loss, accuracy, precision, recall, f1

def train_model(model, criterion, optimizer, train_data, val_data, num_epochs=200):
    best_acc = 0
    scaler = GradScaler()  # 初始化 GradScaler

    Train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    Val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(Train_dataloader, dynamic_ncols=True, ascii=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 全精度运算
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # 使用 autocast 来运行前向传播

            # 半精度运算
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 使用 scaler 来缩放 loss，使得反向传播可以在不溢出的情况下使用半精度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_data)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, Val_dataloader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'mia-others-USLVFM-SSL-busi-3c-best_model.pth')
        print("The best acc is {:4f}".format(best_acc))

# Initialize model
# model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=8) #usfm
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=8)
if Pretrained and os.path.isfile(Pretrained_path):
    print(f"=> loading checkpoint '{Pretrained}'")
    checkpoint = torch.load(Pretrained_path, map_location="cpu")

    state_dict = checkpoint.get('state_dict', checkpoint)  # 支持直接的state_dict或包装过的checkpoint
    new_state_dict = {}


    # 例如，假设我们不想加载最后的线性层权重
    linear_keyword = 'head'  # 这需要根据你模型的具体情况来设定

    for k in list(state_dict.keys()):
        if k.startswith('module.base_encoder') and not k.startswith(f'module.base_encoder.{linear_keyword}'):
            new_key = k.replace('module.base_encoder.', '')  # 删除前缀以匹配模型中的命名
            new_state_dict[new_key] = state_dict[k]

    # 加载处理过的权重到模型的encoder部分
    # net.load_state_dict(new_state_dict, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print('Loaded pretrained(encoder) ----->', Pretrained_path)
else:
    print('=> no pretrained for net')
model.to(device)
# print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)


train_model(model, criterion, optimizer, busi_train_dataset, busi_val_dataset, num_epochs=200)
