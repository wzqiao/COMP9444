# import os
# from torch.utils.data import Dataset
# from PIL import Image

# class InsectDataset(Dataset):
#     def __init__(self, txt_file, root_dir, transform=None, limit=None):
#         self.data = []
#         self.root_dir = root_dir
#         self.transform = transform
#         with open(txt_file, 'r') as file:
#             lines = file.readlines()
#             if limit:
#                 lines = lines[:limit]  # 限制加载的样本数
#             for line in lines:
#                 img_name, label = line.strip().split()
#                 self.data.append((img_name, int(label)))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name, label = self.data[idx]
#         img_path = os.path.join(self.root_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

import os
from torch.utils.data import Dataset
from PIL import Image

class InsectDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, limit=None, label_mapping=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.original_labels = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            if limit:
                lines = lines[:limit]  # 限制加载的样本数
            for line in lines:
                img_name, label = line.strip().split()
                label = int(label)
                self.data.append((img_name, label))
                self.original_labels.append(label)
        
        if label_mapping is not None:
            # 使用提供的标签映射
            self.label_mapping = label_mapping
        else:
            # 创建从原始标签到连续整数的映射
            unique_labels = sorted(set(self.original_labels))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(self.label_mapping)
        print(f"总类别数: {self.num_classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 将原始标签映射到新的连续标签
        label = self.label_mapping[label]
        return image, label
