
> ⚠️ This project also has an English version: [README_EN.md](README_EN.md)
## 🚀 项目简介
基于 Hugging Face 上的 `bert-base-chinese` 和 `chinese-bert-wwm` 模型，完成中文命名实体识别任务，理解序列标注模型的构建逻辑与评估指标。

### 实现模块
- **数据加载与预处理**（BIO 标注格式）  
- **模型设计与训练流程**  
- **模型评估与结果可视化**（可视化使用 `swanlab`）  
- **(新增功能)** 同时支持 `chinese-bert-wwm` 和 `bert-base-chinese` 两个预训练权重，同时支持 `Weibo_NER` 和 `MSRA_NER` 两个数据集（可继续扩展），具体方式为使用 `argparse` 传参或者直接修改 `config.json` 文件。

## ✅ 实验结果
评价指标包括 **F1**、**Precision**、**Recall**，测试结果包含在 `dev.txt` 和 `test.txt` 上，详见 `result/training_log.txt`。

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
│
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
│ ├── bert-base-chineseformsra_NER.pth
│ ├── bert-base-chineseforWeibo_NER.pth
│ ├── chinese-bert-wwmformsra_NER.pth
│ ├── chinese-bert-wwmforWeibo_NER.pth
│ ├── training_log.txt
│ 
│── data_process.py
│── model.py
│── train.py
│── predict.py
├── requirements.txt
│── config.json
│── Config.py
├── data_split_tools.py
│── downloadmodel.py
│── tools.py
└── README.md
```