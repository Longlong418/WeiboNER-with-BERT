from data_process import MyDataset
from transformers import BertTokenizer
from model import BertCnnNER,BertNER
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
#训前准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
max_length = 128
num_epochs = 10

run = swanlab.init(
    # 设置项目
    project="NER",
    # 跟踪超参数与实验元数据
    config={  
        "model":"BERTCNN" ,
        "learning_rate": 2e-5,
        "epochs": num_epochs,
        "dataset": "weibo_ner",
        "batch_size" :batch_size

    },
)


base_dir = os.path.dirname(__file__)  # 获取当前脚本所在路径
train_file_path = os.path.join(base_dir, "data", "train.txt")
val_file_path= os.path.join(base_dir, "data", "dev.txt")
train_df,set_label= MyDataset.data_read(train_file_path)
val_df,_= MyDataset.data_read(val_file_path)



train_dataset = MyDataset(data=train_df)
val_dataset = MyDataset(data=val_df)

num_classes = len(MyDataset.idx_2_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=MyDataset.collate_fn_2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=MyDataset.collate_fn_2)

model = BertCnnNER(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
best_val_f1 = 0.0
save_path = "result/BertCnnNER_model.pth"


if __name__=="__main__":
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_text, batch_label in tqdm(train_loader):
            
            input_ids = batch_text.to(device)
            labels = batch_label.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids)
            outputs=outputs.view(-1, num_classes)  # (batch_size*seq_length, num_classes)
            labels = labels.view(-1)  # (batch_size*seq_length)
            #print(outputs.shape, labels.shape, labels.dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_text, batch_label in tqdm(val_loader):
                input_ids = batch_text.to(device)
                labels = batch_label.to(device)

                outputs = model(input_ids=input_ids)
                preds=torch.argmax(outputs, dim=-1).cpu().numpy()
                labels=labels.cpu().numpy() #labels.shape=(batch_size, seq_len)
                #preds.shape=(batch_size, seq_len, num_classes)
                for i in range(preds.shape[0]):#遍历每个句子的每个token的预测结果
                    true_labels = []
                    pred_labels = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:
                            true_labels.append(MyDataset.idx_2_labels[labels[i][j]])
                            pred_labels.append(MyDataset.idx_2_labels[preds[i][j]])
                    all_labels.append(true_labels)
                    all_preds.append(pred_labels)

        val_f1 =f1_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)
        val_report = classification_report(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        print(val_report)

        #保存每轮的指标
        result_path = os.path.join(base_dir, "result", "training_log.txt")
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n')
            f.write(val_report + '\n')
        
        swanlab.log({"F1": val_f1 ,"loss": avg_loss,\
                    "Precision": val_precision,"Recall": val_recall}, step=epoch)

        # 保存最终的模型
        torch.save(model.state_dict(), save_path)
        print("Finally model saved!")

    swanlab.finish()
    print("Training finished. ")




