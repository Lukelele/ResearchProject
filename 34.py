import torch
from torch.utils.data import Dataset, DataLoader

class TorchDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机的空间坐标 (x, y) 和时间戳 (t)
        x = torch.randn(1).item()  # 随机生成一个 x 坐标
        y = torch.randn(1).item()  # 随机生成一个 y 坐标
        t = torch.normal(mean=0, std=15e-12, size=(1,)).item()  # 随机生成时间戳，标准差为 15 ps
        return (x, y, t)

# 创建数据集实例
num_samples = 1000  # 模拟 1000 个样本
torch_dataset = TorchDataset(num_samples)

# 使用 DataLoader 加载数据
data_loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

# 示例：迭代加载批次数据
for batch in data_loader:
    x_batch, y_batch, t_batch = batch
    print(f"x: {x_batch}, y: {y_batch}, t: {t_batch}")
