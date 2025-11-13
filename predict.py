from data_process import MyDataset
from model import BertCnnNER
import torch
import swanlab
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report,f1_score,precision_score,recall_score
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tools import extract_entities,my_Precision,my_Recall,my_f1_score
from Config import Config
from datetime import datetime
import argparse
# 创建解析器
parser = argparse.ArgumentParser(description="Predict NER")

parser.add_argument("--model_name", type=str, default=None, help="模型名称")
parser.add_argument("--data_name", type=str, default=None, help="数据集名称")
parser.add_argument("--batch_size", type=int, default=None, help="训练批量大小")

args = parser.parse_args()

config=Config()
model_name = args.model_name if args.model_name else "chinese-bert-wwm"
data_name = args.data_name if args.data_name else "msra_NER"
batch_size = args.batch_size if args.batch_size else config.batch_size

device = config.device
model_path=config.models[model_name]
trained_save_root_path=config.trained_save_root_path
data_name="msra_NER"
train_file_path=config.data[data_name][0]
testdata_path= config.data[data_name][2]
model_save_path = os.path.join(trained_save_root_path, f"{model_name}for{data_name}.pth")
val_df,_= MyDataset.data_read(testdata_path)

val_dataset = MyDataset(data=val_df,model_path=model_path,file_path=train_file_path)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=val_dataset.collate_fn_2)

num_classes = len(MyDataset.idx_2_labels)

model = BertCnnNER(num_classes=num_classes,config=config,model_name=model_name).to(device)

model.load_state_dict(torch.load(model_save_path))

if __name__=="__main__":
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(val_loader):
                input_ids = batch_text.to(device)
                labels = batch_label.to(device)
    
                outputs = model(input_ids=input_ids)
                #outputs.shape=(batch_size, seq_len, num_classes)
                preds=torch.argmax(outputs, dim=-1).cpu().numpy()
                #preds.shape=(batch_size, seq_len)#每个seqlen的每个token的预测类别id
                labels=labels.cpu().numpy() #labels.shape=(batch_size, seq_len)
                #labels.shape=(batch_size, seq_len)#每个seqlen的每个token的真实类别id
                for i in range(preds.shape[0]):#遍历每个句子的每个token的预测结果
                    true_labels = []
                    pred_labels = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:
                            true_labels.append(MyDataset.idx_2_labels[labels[i][j]])
                            pred_labels.append(MyDataset.idx_2_labels[preds[i][j]])
                    all_labels.append(true_labels)
                    all_preds.append(pred_labels)
        val_precision = my_Precision(all_labels, all_preds)
        val_recall = my_Recall(all_labels, all_preds)
        val_f1 =my_Recall(all_labels, all_preds)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        
        result_path = os.path.join(trained_save_root_path, "training_log.txt")
        
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f"模型：{model_name} 数据集: {data_name}\n")
            f.write(f"[{timestamp}]\nVal F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n\n")
           


    