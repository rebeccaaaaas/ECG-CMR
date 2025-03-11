import torch
from torch.utils.data import DataLoader
import numpy as np

from model import Generator, CustomDataset

print("start test")
# 设定设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("1")
# 加载已经训练好的生成器模型
generator = Generator().to(device)
generator.load_state_dict(torch.load("/mnt/data2/shitaiyu/data_ecg_cmr/checkpoints/generator9_epoch_318.pt"))
generator.eval()  # 将模型设置为评估模式
print(f"generator loaded")

# 创建测试数据集和数据加载器
test_data_path = "../../data/ECG_CMR/test_data_dict_v7.pt"
test_dataset = CustomDataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 预测并保存结果
test_results = []
with torch.no_grad():  # 关闭梯度计算
    for i, data in enumerate(test_loader):
        ecg_data = data[0].to(device)
        output = generator(ecg_data)
        test_results.append(output.cpu())
        if i % 10 == 0:
            print(f"{i} generate")

test_results = torch.cat(test_results, 0)
torch.save(test_results, "./data/test_results9.pt")

# np.save("test_results.npy", np.array(test_results))  # 保存到文件