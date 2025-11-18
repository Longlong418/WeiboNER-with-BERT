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
from tools import my_f1_score,my_Precision,my_Recall
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from My_Config import My_Config
from datetime import datetime






def evaluate(model,config,val_file_path):
        device = config.device
        model.eval()
        all_preds = []
        all_labels = []
        val_df,_= MyDataset.data_read(val_file_path)
        val_dataset = MyDataset(data=val_df,config=config)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,collate_fn=val_dataset.collate_fn_2)
        with torch.no_grad():
            for batch_text, batch_label in tqdm(val_loader):
                input_ids = batch_text.to(device)
                labels = batch_label.to(device)
                outputs = model(input_ids=input_ids)
                preds=torch.argmax(outputs, dim=-1).cpu().numpy()
                labels=labels.cpu().numpy() #labels.shape=(batch_size, seq_len)
                #preds.shape=(batch_size, seq_len)
                for i in range(preds.shape[0]):#遍历每个句子的每个token的预测结果
                    true_labels = []
                    pred_labels = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:
                            true_labels.append(MyDataset.idx_2_labels[labels[i][j]])
                            pred_labels.append(MyDataset.idx_2_labels[preds[i][j]])
                    all_labels.append(true_labels)
                    all_preds.append(pred_labels)

        val_f1 =my_f1_score(all_labels, all_preds)
        val_precision = my_Precision(all_labels, all_preds)
        val_recall = my_Recall(all_labels, all_preds)

        return val_f1,val_precision,val_recall

def train(model,config): #每一个epoch结束之后会在dev.txt上验证一下
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_path = os.path.join(config.trained_save_root_path, "training_log.txt")
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"模型：{config.model_name} 数据集: {config.data_name}\n")
        f.write(f"[{timestamp}] \n")
        run = swanlab.init(
            # 设置项目
            project="NER",
            # 跟踪超参数与实验元数据
            config={  
                "model_name":config.model_name ,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "dataset": config.data_name,
                "batch_size" :config.batch_size

            },
        )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_file_path=config.train_path
    train_df,set_label= MyDataset.data_read(train_file_path)
    train_dataset = MyDataset(data=train_df,config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,collate_fn=train_dataset.collate_fn_2)
    device = config.device

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch_text, batch_label in tqdm(train_loader,desc=f"Epoch:{epoch+1}/{config.epochs}:"):
            input_ids = batch_text.to(device)
            labels = batch_label.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids)

            num_classes = len(MyDataset.idx_2_labels)
            outputs=outputs.view(-1, num_classes)  # (batch_size*seq_length, num_classes)
            labels = labels.view(-1)  # (batch_size*seq_length)
            #print(outputs.shape, labels.shape, labels.dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        #评估
        val_file_path=config.dev_path
        val_f1,val_precision,val_recall=evaluate(model=model,config=config,val_file_path=val_file_path)

        print(f'Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

        #保存每轮的指标
        result_path = os.path.join(config.trained_save_root_path, "training_log.txt")
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n')
        swanlab.log({"F1": val_f1 ,"loss": avg_loss,\
                    "Precision": val_precision,"Recall": val_recall}, step=epoch)

    # 保存最终的模型
    save_path = os.path.join(config.trained_save_root_path, f"{config.model_name}_for_{config.data_name}.pth")
    torch.save(model.state_dict(), save_path)
    config.trained_model_path = str(save_path)  
    config.save()
    print("Finally model saved!")
    swanlab.finish()
    print("Training finished. ")
    

if __name__=="__main__":
    
    train()




