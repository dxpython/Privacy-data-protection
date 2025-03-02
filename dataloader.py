import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        :param csv_file: 包含元数据的CSV文件路径
        :param img_dir: 图像数据所在的文件夹路径
        :param transform: 可选的图像转换
        """
        self.data_frame = pd.read_csv(csv_file)  # 读取CSV文件
        self.img_dir = img_dir  # 图像文件夹路径
        self.transform = transform  # 图像转换操作

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """根据索引返回图像和标签"""
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 1] + '.jpg')  
        image = Image.open(img_name).convert("RGB")  
        
        # 获取标签
        label = self.data_frame.iloc[idx, 3] 
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# 创建Dataset对象
dataset = SkinCancerDataset(csv_file='/root/autodl-tmp/HE/dataset/train-metadata.csv',
                             img_dir='/root/autodl-tmp/HE/dataset/train-image/image',
                             transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 测试code
for images, labels in dataloader:
    print(images.shape, labels.shape) 
    break  
