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
from Config import Config
from datetime import datetime
import argparse
# 创建解析器
parser = argparse.ArgumentParser(description="Train NER")

parser.add_argument("--model_name", type=str, default=None, help="模型名称")
parser.add_argument("--data_name", type=str, default=None, help="数据集名称")
parser.add_argument("--batch_size", type=int, default=None, help="训练批量大小")
parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
parser.add_argument("--lr", type=float, default=None, help="学习率")
args = parser.parse_args()
#参数配置
config=Config()
#常改的参数
model_name = args.model_name if args.model_name else "chinese-bert-wwm"
data_name = args.data_name if args.data_name else "msra_NER"
batch_size = args.batch_size if args.batch_size else config.batch_size
num_epochs = args.epochs if args.epochs else config.epochs
lr = args.lr if args.lr else config.learing_rate
model_path=config.models[model_name]
device = config.device
trained_save_root_path = config.trained_save_root_path

save_path = os.path.join(trained_save_root_path, f"{model_name}for{data_name}.pth")

run = swanlab.init(
    # 设置项目
    project="NER",
    # 跟踪超参数与实验元数据
    config={  
        "model_name":model_name ,
        "learning_rate": lr,
        "epochs": num_epochs,
        "dataset": data_name,
        "batch_size" :batch_size

    },
)

#0 1 2表示 train、dev、test
train_file_path=config.data[data_name][0]
val_file_path=config.data[data_name][1]

train_df,set_label= MyDataset.data_read(train_file_path)
val_df,_= MyDataset.data_read(val_file_path)


train_dataset = MyDataset(data=train_df,model_path=model_path,file_path=train_file_path)
val_dataset = MyDataset(data=val_df,model_path=model_path,file_path=train_file_path)

num_classes = len(MyDataset.idx_2_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=train_dataset.collate_fn_2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=val_dataset.collate_fn_2)

model = BertCnnNER(num_classes=num_classes,config=config,model_name=model_name).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Training loop
def train():
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_text, batch_label in tqdm(train_loader,desc=f"Epoch:{epoch+1}/{num_epochs}:"):
            
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

        val_f1 =my_f1_score(all_labels, all_preds)
        val_precision = my_Precision(all_labels, all_preds)
        val_recall = my_Recall(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

        #保存每轮的指标
        result_path = os.path.join(trained_save_root_path, "training_log.txt")
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n')
        swanlab.log({"F1": val_f1 ,"loss": avg_loss,\
                    "Precision": val_precision,"Recall": val_recall}, step=epoch)

        # 保存最终的模型
    torch.save(model.state_dict(), save_path)
    print("Finally model saved!")

    swanlab.finish()
    print("Training finished. ")
    

if __name__=="__main__":
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_path = os.path.join(trained_save_root_path, "training_log.txt")
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"模型：{model_name} 数据集: {data_name}\n")
        f.write(f"[{timestamp}] \n")
    train()




