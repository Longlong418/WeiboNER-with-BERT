> ⚠️ This project also has a Chinese version: [README.md](README.md)

## 🚀 Project Overview
Based on Hugging Face's `bert-base-chinese` and `chinese-bert-wwm` models, this project performs Chinese Named Entity Recognition (NER) tasks, helping users understand the logic of sequence labeling model construction and evaluation metrics.

### Implemented Modules
- **Data Loading and Preprocessing** (BIO labeling format)  
- **Model Design and Training Pipeline**  
- **Model Evaluation and Result Visualization** (visualization uses `swanlab`)  

## ✅ Experimental Results
Evaluation metrics include **F1**, **Precision**, and **Recall**.  
The following results are on `test.txt`:

| Dataset | Model                | Precision | Recall | F1 Score |
|---------|--------------------|-----------|--------|----------|
| weibo   | bert-base-chinese   | 0.6284    | 0.6634 | 0.6455   |
| weibo   | chinese-bert-wwm    | 0.6117    | 0.6828 | 0.6453   |
| msra    | bert-base-chinese   | 0.9431    | 0.9429 | 0.9430   |
| msra    | chinese-bert-wwm    | 0.9365    | 0.9365 | 0.9365   |

Results on `dev.txt` can be found in `result/training_log.txt`.

## How to Run the Code

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2.Train the Model
- --mode: run mode, set to train for training

- --config_path: path to the configuration file, optional (default: ./NER_Config/Bertbase_Weibo_Config.json)
Each config file corresponds to a single experiment.

```bash
python main.py --mode train --config_path ./NER_Config/Bertbase_Weibo_Config.json

```
After training, the model weights will be saved in the directory specified by trained_save_root_path in the config file, and the file path will also be recorded in the JSON.

### 3. Evaluate on the Test Set
```bash
python main.py --mode eval --config_path ./NER_Config/Bertbase_Weibo_Config.json

```

The script will output F1, Precision, Recall, and record the results in training_log.txt.

### 4.Predict Entities in a Single Sentence
```bash
python main.py --mode predict --config_path ./NER_Config/Bertbase_Weibo_Config.json

```

The program will prompt you to input a Chinese sentence.
It will output the entities and their types, e.g.:
```bash

[('小明', 'PER.NAM'), ('北京', 'GPE.NAM'), ('大学', 'ORG.NOM')]

```
![](https://img.xlonglong.cn/img/202511181811635.png)


## 📊 Dataset Sources

1. **Weibo NER Dataset**
    - Description: This dataset includes training (1350), validation (269), and test (270) sets. Entity types include geopolitical entities (GPE.NAM), locations (LOC.NAM), organizations (ORG.NAM), and people (PER.NAM) along with nominal forms (NOM).  
    - Language: Chinese  
    - Train/Dev/Test sizes: 1350/269/270  
    - Number of entity categories: 4  
    - Paper: [https://aclanthology.org/D15-1064.pdf](https://aclanthology.org/D15-1064.pdf)  
    - Download: [https://tianchi.aliyun.com/dataset/144312](https://tianchi.aliyun.com/dataset/144312)  
    - GitHub: [https://github.com/hltcoe/golden-horse](https://github.com/hltcoe/golden-horse)

2. **MSRA NER Dataset**
    - Description: The MSRA dataset is a Chinese NER dataset in the news domain, including training (46364) and test (4365) sets. Entity types include locations (LOC), people (NAME), and organizations (ORG).  
    - Language: Chinese  
    - Train/Test sizes: 46364/4365 (The training set was re-split into train and dev sets using `data_split_tools.py`)  
    - Number of entity categories: 3  
    - Paper: [https://aclanthology.org/W06-0115.pdf](https://aclanthology.org/W06-0115.pdf)  
    - Download: [https://tianchi.aliyun.com/dataset/144307](https://tianchi.aliyun.com/dataset/144307)

## 📂 Project Structure Example
> Note: final model weights and pretrained model folders are not uploaded

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
