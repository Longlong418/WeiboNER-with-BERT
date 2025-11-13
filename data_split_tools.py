import os
from sklearn.model_selection import train_test_split



input_file = "./data/msra_NER/train.txt"
train_file ="./data/msra_NER/train_split.txt"
dev_file = "./data/msra_NER/dev_split.txt"
dev_ratio = 0.2  # 验证集比例
random_seed = 42

def write_data(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(data_list))  

with open(input_file, 'r', encoding='utf-8') as f:
    all_data = f.read().strip().split("\n\n") 

print(f"总样本数: {len(all_data)}")

train_data, dev_data = train_test_split(
    all_data, test_size=dev_ratio, random_state=random_seed
)

print(f"训练集样本数: {len(train_data)}, 验证集样本数: {len(dev_data)}")

write_data(train_file, train_data)
write_data(dev_file, dev_data)

print(f"划分完成！\n训练集: {train_file}\n验证集: {dev_file}")
