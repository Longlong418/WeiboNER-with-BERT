## 🚀 Project Overview
Based on the `bert-base-chinese` and `chinese-bert-wwm` models from Hugging Face, this project performs Chinese Named Entity Recognition (NER), helping to understand the logic of sequence labeling models and evaluation metrics.  

Implemented modules:
- Data loading and preprocessing (BIO annotation format)
- Model design and training workflow
- Model evaluation and result visualization (visualization using swanlab)
- (New) Supports both `chinese-bert-wwm` and `bert-base-chinese` pre-trained weights, and supports both Weibo_NER and MSRA_NER datasets (can be further extended). Users can select datasets/models via `argparse` arguments or by modifying the `config.json` file.

## ✅ Experimental Results
Evaluation metrics include F1, Precision, and Recall.  
Test results are available on `dev.txt` and `test.txt`.  
See `result/training_log.txt` for details.

## 📊 Dataset Sources
1. **Weibo NER Dataset**  

    **Description:** This dataset contains a training set (1350), validation set (269), and test set (270). Entity types include geopolitical entities (GPE.NAM), locations (LOC.NAM), organizations (ORG.NAM), persons (PER.NAM), and corresponding nominal references (ending with NOM).  

    **Language:** Chinese  

    **Train/Validation/Test sizes:** 1350/269/270  

    **Number of entity types:** 4  

    **Paper:** [https://aclanthology.org/D15-1064.pdf](https://aclanthology.org/D15-1064.pdf)  

    **Download:** [https://tianchi.aliyun.com/dataset/144312](https://tianchi.aliyun.com/dataset/144312)  

    **GitHub:** [https://github.com/hltcoe/golden-horse](https://github.com/hltcoe/golden-horse)

2. **MSRA NER Dataset**  

    **Description:** The MSRA dataset is a Chinese NER dataset in the news domain. It contains a training set (46364) and a test set (4365). Entity types include locations (LOC), persons (NAME), and organizations (ORG).  

    **Language:** Chinese  

    **Train/Test sizes:** 46364/4365 (I re-split the training set into training and validation sets using `data_split_tools.py`)  

    **Number of entity types:** 3  

    **Paper:** [https://aclanthology.org/W06-0115.pdf](https://aclanthology.org/W06-0115.pdf)  

    **Download:** [https://tianchi.aliyun.com/dataset/144307](https://tianchi.aliyun.com/dataset/144307)  

## 📂 Project Structure Example
**Note:** Final trained models and pre-trained model folders are not uploaded.
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
└── README_EN.md
```
