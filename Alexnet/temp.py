import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from datagenerator import InsectDataset  # 确保导入正确的 InsectDataset 类
from collections import Counter
import torch.nn as nn
from utils import train
from utils import plot_history
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # 确保使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行的设备是: {device}")

    # # 定义数据集和预处理
    # transform = transforms.Compose([
    #     transforms.Resize((112, 112)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    # ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    # # 初始化数据集
    # print("加载数据集...")
    # dataset = InsectDataset('D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/train.txt', 'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/images', transform=transform,limit=10000)

    # # 提取标签
    # labels = [label for _, label in dataset]

    # # 计算类别权重
    # num_classes = num_classes = len(set(labels))#102
    # label_counts = Counter(labels)
    # print(f"标签计数: {label_counts}")
    # class_weights = torch.tensor([1.0 / label_counts.get(i, 0) if label_counts.get(i, 0) > 0 else 0.0 
    #                             for i in range(num_classes)], dtype=torch.float).to(device)

    # 加载训练集
    train_dataset = InsectDataset(
        'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/train.txt',
        'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/images',
        transform=transform,
    )
        # 从训练集获取标签映射和类别数
    label_mapping = train_dataset.label_mapping
    num_classes = train_dataset.num_classes

    # 加载测试集，使用与训练集相同的标签映射
    test_dataset = InsectDataset(
        'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/test.txt',
        'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/images',
        transform=transform,
        label_mapping=label_mapping
    )

    # 加载验证集，使用与训练集相同的标签映射
    val_dataset = InsectDataset(
        'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/val.txt',
        'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/images',
        transform=transform,
        label_mapping=label_mapping
    )




    class AlexNet(nn.Module):  # 训练 ALexNet
        '''
        5层卷积，3层全连接
        ''' 
        def __init__(self):
            super(AlexNet, self).__init__()
            # 五个卷积层 输入 32 * 32 * 3
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),   # (32-3+2)/1+1 = 32
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
            )
            self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
            )
            self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (8-3+2)/1+1 = 8
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
            )
            self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (4-3+2)/1+1 = 4
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
            )
            self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# (2-3+2)/1+1 = 2
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
            )                            # 最后一层卷积层，输出 1 * 1 * 128
            # 全连接层
            self.dense = nn.Sequential(
                nn.Linear(6272, 120),#112 112. 128*3*3
                nn.ReLU(),
                nn.Dropout(0.3),  # 添加 Dropout
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Dropout(0.3),  # 添加 Dropout
                nn.Linear(84, num_classes)
            )
            self._initialize_weights()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            #print(f"卷积层输出形状: {x.shape}")  # 检查输出形状
            x = x.view(x.size()[0],-1)
            x = self.dense(x)
            return x
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)






    # 定义模型、优化器和损失函数
    net = AlexNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.25):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, inputs, targets):
            BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            return F_loss

    criterion = FocalLoss()


    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)
    # 使用 StepLR 调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



    # # 创建数据加载器
    # train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

    # 训练模型
    num_epochs = 30  # 增加训练轮数
    best_val_acc = 0.0  # 用于保存最佳模型
    patience = 5  # 早停的耐心值
    trigger_times = 0  # 早停计数器

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练集上的准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # 在验证集上评估模型
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        # 学习率调度器步进（如果使用 ReduceLROnPlateau）
        # scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Training Loss: {running_loss/len(train_loader):.4f}, "
            f"Training Accuracy: {train_acc:.2f}%, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), 'best_model.pth')
            trigger_times = 0  # 重置早停计数器
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # 加载测试集
    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    test_dataset = InsectDataset('D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/test.txt', 'D:/9444Project/Classification/ip102_v1.1/ip102_v1.1/images', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 在测试集上评估模型
    net.eval()
    net.load_state_dict(torch.load('model.pth'))

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算准确率
    correct = sum(np.array(all_predictions) == np.array(all_labels))
    total = len(all_labels)
    test_acc = 100 * correct / total
    print(f'测试准确率: {test_acc:.2f}%')

    # 生成分类报告
    report = classification_report(all_labels, all_predictions, digits=4)
    print("分类报告:")
    print(report)

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("混淆矩阵:")
    print(conf_matrix)

    # # 可视化混淆矩阵
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues')
    # plt.xlabel('预测标签')
    # plt.ylabel('真实标签')
    # plt.title('混淆矩阵')
    # plt.show()

