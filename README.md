
## 🚀 项目简介
基于 bert-base-chinese 模型，完成中文实体识别任务，理解序列标注模型的构建逻辑与评估指标。
实现模块：
- 数据加载与预处理（BIO标注格式）
- 模型设计与训练流程 
- 模型评估与结果可视化(可视化使用swanlab)

调用seqeval库测出来的结果为：
Val F1: 0.6448,            Precision: 0.6103, Recall: 0.6835

自己手写评测结果为：
Val F1: 0.6706,            Precision: 0.6490, Recall: 0.6938

## 数据集来源：
weibo命名实体识别数据集

简介：本数据集包括训练集（1350）、验证集（269）、测试集（270），实体类型包括地缘政治实体(GPE.NAM)、地名(LOC.NAM)、机构名(ORG.NAM)、人名(PER.NAM)及其对应的代指(以NOM为结尾)。

语种：Chinese

"训练集/验证集/测试集"数量: 1350/269/270

实体类别数量：4

论文：https://aclanthology.org/D15-1064.pdf

下载地址：https://tianchi.aliyun.com/dataset/144312

Github: https://github.com/hltcoe/golden-horse

## 📂 项目结构示例
注：data、result文件夹没有上传

```
Weibo-NER/
│
├── data/ 
│ ├── train.txt
│ ├── dev.txt
│ └── test.txt
│ └── class.txt
├── result/ #实验结果
│ ├── BertCnnNER_model.pth
│ ├── training_log.txt
│ 
│── data_process.py
│── model.py
│── train.py
│── predict.py
├── requirements.txt
└── README.md
```