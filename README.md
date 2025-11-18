
> ⚠️ This project also has an English version: [README_EN.md](README_EN.md)
## 🚀 项目简介
基于 Hugging Face 上的 `bert-base-chinese` 和 `chinese-bert-wwm` 模型，完成中文命名实体识别任务，理解序列标注模型的构建逻辑与评估指标。

### 实现模块
- **数据加载与预处理**（BIO 标注格式）  
- **模型设计与训练流程**  
- **模型评估与结果可视化**（可视化使用 `swanlab`）  

## ✅ 实验结果
评价指标包括 **F1**、**Precision**、**Recall**，
以下为test.txt上的结果：
| 数据集 | 模型                | Precision | Recall | F1值    |
|--------|-------------------|-----------|--------|---------|
| weibo  | bert-base-chinese  | 0.6284     | 0.6634  |  0.6455   |
| weibo  | chinese-bert-wwm   | 0.6117     | 0.6828 | 0.6453   |
| msra   | bert-base-chinese  | 0.9431    | 0.9429 | 0.9430   |
| msra   | chinese-bert-wwm   | 0.9365     | 0.9365  | 0.9365   |

dev.txt结果详见 `result/training_log.txt`。

## 如何运行代码
### 1. 安装依赖

```bash
pip install -r requirements.txt
```
### 2.加载预训练模型到本地
```bash
python download_model.py
```
预训练模型会保存到`./model`文件夹下

### 3.训练模型
--mode：运行模式，训练时设置为 train
--config_path：配置文件路径，可选，默认路径为 ./NER_Config/Bertbase_Weibo_Config.json
一个配置文件对应一个实验
```bash
python main.py --mode train --config_path ./NER_Config/Bertbase_Weibo_Config.json #这里可以替换为你想要的json配置文件 
```
训练完成后，模型权重会保存在配置文件中 trained_save_root_path 指定的目录下，对应文件路径会保存在json文件中

### 4.在测试集上评估
```bash

python main.py --mode eval --config_path ./NER_Config/Bertbase_Weibo_Config.json 

```
脚本会输出 F1、Precision、Recall，并将结果记录到 training_log.txt

### 5.对单条句子进行实体预测
```bash
python main.py --mode predict --config_path ./NER_Config/Bertbase_Weibo_Config.json
```
运行后会提示输入一句中文文本
程序会输出该句子中的实体及对应类型，例如：
```bash
[('小明', 'PER.NAM'), ('北京', 'GPE.NAM'), ('大学', 'ORG.NOM')]
```
![](https://img.xlonglong.cn/img/202511181811635.png)

## 📊 数据集来源

1. **Weibo 命名实体识别数据集**
    - 简介：本数据集包括训练集（1350）、验证集（269）、测试集（270），实体类型包括地缘政治实体 (GPE.NAM)、地名 (LOC.NAM)、机构名 (ORG.NAM)、人名 (PER.NAM) 及其代指 (以 NOM 结尾)。  
    - 语种：Chinese  
    - "训练集/验证集/测试集" 数量：1350/269/270  
    - 实体类别数量：4  
    - 论文：[https://aclanthology.org/D15-1064.pdf](https://aclanthology.org/D15-1064.pdf)  
    - 下载地址：[https://tianchi.aliyun.com/dataset/144312](https://tianchi.aliyun.com/dataset/144312)  
    - Github: [https://github.com/hltcoe/golden-horse](https://github.com/hltcoe/golden-horse)

2. **MSRA 命名实体识别数据集**
    - 简介：MSRA 数据集是面向新闻领域的中文命名实体识别数据集，包括训练集（46364）、测试集（4365），实体类型包括地名 (LOC)、人名 (NAME)、组织名 (ORG)。  
    - 语种：Chinese  
    - "训练集/测试集" 数量：46364/4365（我使用脚本在训练集上重新划分了训练集和验证集，详见 `data_split_tools.py`）  
    - 实体类别数量：3  
    - 论文：[https://aclanthology.org/W06-0115.pdf](https://aclanthology.org/W06-0115.pdf)  
    - 下载地址：[https://tianchi.aliyun.com/dataset/144307](https://tianchi.aliyun.com/dataset/144307)

## 📂 项目结构示例
> 注：最终模型以及预训练模型文件夹未上传

```
Weibo-NER/
├── model/ 
│ ├── models--bert-base-chinese
│ ├── models--hfl--chinese-bert-wwm
├── data/ 
│ ├── msra_NER
│   ├── train.txt #原始数据
│   ├── train_split.txt
│   ├── dev_split.txt
│   ├── test.txt
│   ├──msra_ner.json
│ ├──Weibo_NER
│   ├── train.txt
│   ├── test.txt
│   ├── dev.txt
│   ├── class.txt
├── result/ #实验结果
│ ├── bert-base-chinese_for_Weibo_NER.pth
│ ├── bert-base-chineseforWeibo_NER.pth
│ ├── chinese-bert-wwmformsra_NER.pth
│ ├── chinese-bert-wwm_for_Weibo_NER.pth
│ ├── training_log.txt
├── NER_config/
│ ├── Bertbase_msra_Config.json
│ ├── Bertbase_Weibo_Config.json
│ ├── Chinese_bert_wwm_msra_config.json.json
│ ├── Chinese_bert_wwm_Weibo_config.json.json
│ 
│── data_process.py
│── main.py
│── model.py
│── train_evaluate.py
├── requirements.txt
│── My_Config.py
├── data_split_tools.py
│── downloadmodel.py
│── tools.py
└── README.md
└── README_EN.md
```

