import numpy as np


np.random.seed(42)  # 设置随机种子以确保结果可重现

test_embedding = np.random.randn(512).tolist()  # 生成一个512维的随机向量并将array数组转换成python列表
print(test_embedding)
print(f"向量维度: {len(test_embedding)}")