import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from data_process import  MyDataset
from transformers import BertModel, AutoTokenizer
class BertCnnNER(nn.Module):
    def __init__(self,num_classes,dropout=0.1):
        super(BertCnnNER, self).__init__()
        self.bert = BertModel.from_pretrained('C:\\Users\jd\\.cache\\huggingface\\hub\\models--bert-base-chinese\\snapshots\\8f23c25b06e129b6c986331a13d8d025a92cf0ea')
        hidden_size = self.bert.config.hidden_size  # BERT的隐藏层大小，通常为768
        # 因为1D卷积的输出长度计算公式为：(L_in + 2*padding - dilation*(kernel_size-1) -1)/stride +1# 这里stride=1,dilation=1
        # =>(L_in + 2*padding - (kernel_size-1) -1) +1 = L_in + 2*padding - kernel_size +1
        # padding=(kernel_size-1)/2时，L_out=L_in，且只能对于kernel_size为奇数时有效
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=100, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=100, kernel_size=5,padding=2)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=100, kernel_size=7,padding=3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3*100, num_classes)#  3个卷积层的输出通道数之和
    def forward(self, input_ids):
        outputs=self.bert(input_ids=input_ids,
                          attention_mask=(input_ids!=0),#padding部分mask掉
                          )
        x=outputs.last_hidden_state #(batch_size, seq_length, hidden_size)
        x = x.permute(0, 2, 1)#(batch_size, seq_length, hidden_size)->(batch_size, hidden_size, seq_length)
                              #conv1d的输入为(batch_size, in_channels, seq_length),这里将hidden_size对齐到in_channels
        x1 = F.relu(self.conv1(x))  # (batch_size, out_channels, L_out1)
        x2 = F.relu(self.conv2(x))  # (batch_size, out_channels, L_out2)
        x3 = F.relu(self.conv3(x))  # (batch_size, out_channels, L_out3)
   
        x = torch.cat((x1, x2, x3), dim=1) #拼接 (batch_size, out_channels*3,L_out)
        x=x.permute(0,2,1) #(batch_size,L_out,out_channels*3) 这里L_out=L_in=seq_length=输入序列长度
        x = self.dropout(x)                 
        logits = self.fc(x)   #得到(batch_size,sel_length,num_classes)

        return logits
    
class BertNER(nn.Module):
    def __init__(self,num_classes,dropout=0.1):
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained('C:\\Users\jd\\.cache\\huggingface\\hub\\models--bert-base-chinese\\snapshots\\8f23c25b06e129b6c986331a13d8d025a92cf0ea')
        hidden_size = self.bert.config.hidden_size  # BERT的隐藏层大小，通常为768
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)#  3个卷积层的输出通道数之和
    def forward(self, input_ids):
        outputs=self.bert(input_ids=input_ids,
                          attention_mask=(input_ids!=0),#padding部分mask掉
                          )
        x=outputs.last_hidden_state #(batch_size, seq_length, hidden_size)
        x = self.dropout(x)                 
        logits = self.fc(x)   #得到(batch_size,sel_length,num_classes)

        return logits


