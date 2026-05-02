#print("hello")

# import torch

# import numpy
# import stable_baselines3
# print(f'NumPy 版本: {numpy.__version__}')
# print(f'SB3 版本: {stable_baselines3.__version__}')
# print('环境验证通过')

# print("PyTorch版本:", torch.__version__)
# print("CUDA是否可用:", torch.cuda.is_available())  # 必须返回 True
# print("PyTorch编译的CUDA版本:", torch.version.cuda)
# print("GPU型号:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")



import stable_baselines3 as sb3
import torch
import os

# 1. 读取高版本 NumPy 下保存的模型
model_path = "D:/AAAAA/combat/combat_train/combat_train/models/final_model_lzj_2m.zip"
print("正在高版本环境中加载模型...")
model = sb3.PPO.load(model_path)

# 2. 剥离 NumPy 外壳，只提取核心的 PyTorch 神经网络权重（State Dict）
pure_weights = model.policy.state_dict()

# 3. 保存为纯净的 .pth 权重文件
save_path = "D:/AAAAA/combat/combat_train/combat_train/models/clean_policy_weights.pth"
torch.save(pure_weights, save_path)

print(f"提取成功！纯净权重已保存至: {save_path}")