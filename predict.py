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
"""
y_true, y_pred的格式为：
[['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O'], 
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NAM', ...], 
['O', 'O', 'O', 'O', 'B-ORG.NAM', 'I-ORG.NAM', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], 
...
...
...,
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...]
]
#所有的类型标签为：
['B-GPE.NAM', 'B-GPE.NOM', 'B-LOC.NAM', 'B-LOC.NOM', 'B-ORG.NAM', 'B-ORG.NOM', 'B-PER.NAM', 'B-PER.NOM', 
'I-GPE.NAM', 'I-GPE.NOM', 'I-LOC.NAM', 'I-LOC.NOM', 'I-ORG.NAM', 'I-ORG.NOM', 'I-PER.NAM', 'I-PER.NOM', 'O']

"""
#获取实体
def extract_entities(labels)->list: #return [(seq_idx,entity_type,start_idx,end_idx),...]
    all_entities = []
    for i in range(len(labels)):
        start_idx=None
        end_idx=None
        entity_type=None
        n=len(labels[i])
        j=0
        while j<n:
            if labels[i][j]=='O':
                j+=1
                continue
            if labels[i][j][0]=='B':
                start_idx=j
                entity_type=labels[i][j][2:]
                k=j+1
                while k<n and labels[i][k][0]=='I' and labels[i][k][2:]==entity_type:
                    k+=1
                end_idx=k-1
                all_entities.append((i,entity_type,start_idx,end_idx))
                j=k
                continue
            #凭空的I 不算入实体
            if labels[i][j][0]=='I':
                j+=1
                continue
    return all_entities                                                
# test=[['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O'], 
# ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NAM', 'I-PER.NAM', 'I-PER.NAM', 'O', 'O'], 
# ['O', 'O', 'O', 'O', 'B-ORG.NAM', 'I-ORG.NAM', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ]]

# print(extract_entities(test))
def my_Precision(y_true, y_pred):
    #预测为正的样本中，有多少是真正的正样本
    #转成集合才能进行交集操作
    true_entities=set(extract_entities(y_true))
    pred_entities=set(extract_entities(y_pred))
    tp=true_entities&pred_entities  #实际上预测对的样本
    return len(tp)/(len(pred_entities)+1e-8)

    
def my_Recall(y_true, y_pred):
    #实际为正的样本中，有多少被预测为正样本
    true_entities=set(extract_entities(y_true))
    pred_entities=set(extract_entities(y_pred))
    tp=true_entities&pred_entities  #实际上预测对的样本
    return len(tp)/(len(true_entities)+1e-8)
def my_f1_score(y_true, y_pred):
    #调和平均的意义：当其中一个指标特别低时，整体得分应被显著拉低
    precision=my_Precision(y_true, y_pred)
    recall=my_Recall(y_true, y_pred)
    return 2*precision*recall/(precision+recall+1e-8)


     


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
max_length = 128
num_epochs = 10

base_dir = os.path.dirname(__file__)  # 获取当前脚本所在路径
model_path=os.path.join(base_dir, "result", "BertCnnNER_model.pth")
testdata_path= os.path.join(base_dir, "data", "test.txt")

val_df,_= MyDataset.data_read(testdata_path)

val_dataset = MyDataset(data=val_df)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=MyDataset.collate_fn)

num_classes = len(MyDataset.idx_2_labels)

model = BertCnnNER(num_classes=num_classes).to(device)

model.load_state_dict(torch.load(model_path))

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
        # val_report = classification_report(all_labels, all_preds)
        print(f'Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        # print(val_report)

        result_path = os.path.join(base_dir, "result", "training_log.txt")
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n')
            # f.write(val_report + '\n')


    