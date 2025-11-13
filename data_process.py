import os
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


"""
idx_2_labels为：
['B-GPE.NAM', 'B-GPE.NOM', 'B-LOC.NAM', 'B-LOC.NOM', 'B-ORG.NAM', 'B-ORG.NOM', 'B-PER.NAM', 'B-PER.NOM', 
'I-GPE.NAM', 'I-GPE.NOM', 'I-LOC.NAM', 'I-LOC.NOM', 'I-ORG.NAM', 'I-ORG.NOM', 'I-PER.NAM', 'I-PER.NOM', 'O']
"""
class MyDataset(Dataset):
    train_examples, set_label = None, None  
    idx_2_labels = None     # list sorted labels
    labels_2_idx = None     # dict label -> idx
    tokenizer=None
        
    def __init__(self,data,model_path,file_path=None):#file_path需要传入训练集的path
        self.data=data
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)
        if MyDataset.train_examples is None:
            MyDataset.train_examples, MyDataset.set_label = self.data_read(file_path)
        self.idx_2_labels=sorted(MyDataset.set_label)
        self.labels_2_idx={j:i for i,j in enumerate(self.idx_2_labels)}
        self.model_path=model_path
        #放到静态变量中
        MyDataset.idx_2_labels = self.idx_2_labels
        MyDataset.labels_2_idx = self.labels_2_idx
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        text=self.data[index]['tokens']
        label=self.data[index]['labels']
        return text ,label
    
    @staticmethod
    def data_read(path):
        """
        返回： data为列表，每项为 {'tokens': [...], 'labels': [...]}
        第二项为一个set,表示所有种类
        """
        data = []

        with open(path, 'r', encoding='utf8') as f:
            all_data=f.read().split("\n\n")
            for Data_Item in all_data:
                tokens=[]
                labels=[]
                for i,line_pair in enumerate(Data_Item.split('\n')):
                    token,label=line_pair.split("\t")
                    tokens.append(token)
                    labels.append(label)
                data.append({'tokens': tokens, 'labels': labels})

        set_label = set()
        for temp in data:
            set_label.update(temp['labels'])

        return data,list(set_label)
    
    #投票机制，选最多的的标签作为该token的标签
    def collate_fn(self,batch_data):
        process_inputs=[]
        process_labels=[]
        for batch_text, batch_label in batch_data:
            text_str=''.join(batch_text)
            tokenizer=self.tokenizer
            encoding = tokenizer(
                text_str,
                return_offsets_mapping=True,
                padding=False,
                truncation=False,
                return_tensors=None,
                is_split_into_words=False #因为数据集本身不是划分的很好，所以让tokenizer再分一次词
            )
            input_ids = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]#offset_mapping 是一个 列表，长度等于 tokenizer 输出的 token 数量
            #每个元素是(start, end)，表示这个 token 在原始字符串中的 起止字符位置（半开区间 [start, end)）
            #其中：[CLS] offset = (0,0)，[SEP] offset = (0,0)
            #处理对齐问题
            new_labels = []
            for start,end in offset_mapping:
                if start == end == 0:
                    new_labels.append(-100)#这里将[CLS]和[SEP]的标签设为-1，表示忽略
                    continue
                
                token_labels = []
                for i in range(start,min(end,len(batch_label))):
                    token_labels.append(MyDataset.labels_2_idx.get(batch_label[i], -100))#找不到就取-100，后边训练时会将ignore_index设为-100
                
                #这里的token_labels是当前被tokenizer分出的一个token对应的原始字符的标签索引列表
                if token_labels:
                    label_counts={}
                    for label_idx in token_labels:
                        if label_idx == -100:
                            continue
                        if label_idx in label_counts:
                            label_counts[label_idx]+=1
                        else:
                            label_counts[label_idx]=1
                    #等价于for label_idx in token_labels:
                        # if label_idx != -100:
                            # label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
                    #取出现次数最多的标签作为该token的标签
                    if label_counts:
                        max_temp=0
                        max_label_idx = None
                        for k in label_counts:
                            if label_counts[k]>max_temp:
                                max_temp=label_counts[k]
                                max_label_idx = k
                        #等价于：max_label_idx=max(label_counts, key=label_counts.get)
                        new_labels.append(max_label_idx)
                    else:
                        new_labels.append(-100)
                else:
                    new_labels.append(-100)

            assert len(input_ids) == len(new_labels), "处理异常，序列数量合label数量不匹配"
            
            process_inputs.append(input_ids)
            process_labels.append(new_labels)
        #处理padding 这里策略是补齐到当前batch的最大长度
        last_inputs_ids=[]
        last_labels=[]
        max_len = max(len(seq) for seq in process_inputs)
        if max_len > 512:
            print(f"检测到超长样本，长度={max_len}，自动截断至512")
            process_inputs = [seq[:512] for seq in process_inputs]
            process_labels = [seq[:512] for seq in process_labels]
            max_len = 512
        for input_ids, labels in zip(process_inputs, process_labels):
            padding_length = max_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length #补齐input_ids,该模型为0
            labels += [-100] * padding_length

            last_inputs_ids.append(input_ids)
            last_labels.append(labels)



        batch_text = torch.tensor(last_inputs_ids, dtype=torch.long)
        batch_label = torch.tensor(last_labels, dtype=torch.long)
        return batch_text, batch_label
    
    #选第一个标签作为该token的标签

    def collate_fn_2(self,batch_data):
        process_inputs=[]
        process_labels=[]
        for batch_text, batch_label in batch_data:
            text_str=''.join(batch_text)
            tokenizer=self.tokenizer
            encoding = tokenizer(
                text_str,
                return_offsets_mapping=True,
                padding=False,
                truncation=False,
                return_tensors=None,
                is_split_into_words=False #因为数据集本身不是划分的很好，所以让tokenizer再分一次词
            )
            input_ids = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]#offset_mapping 是一个 列表，长度等于 tokenizer 输出的 token 数量
            #每个元素是(start, end)，表示这个 token 在原始字符串中的 起止字符位置（半开区间 [start, end)）
            #其中：[CLS] offset = (0,0)，[SEP] offset = (0,0)
            
            #处理对齐问题
            new_labels = []
            for start,end in offset_mapping:
                if start == end == 0:
                    new_labels.append(-100)
                    continue
                #取第一个字符的标签作为该token的标签
                if start < len(batch_label):
                    label_idx = MyDataset.labels_2_idx.get(batch_label[start], -100)
                    new_labels.append(label_idx)
                else:
                    new_labels.append(-100)
            
            assert len(input_ids) == len(new_labels), "处理异常，序列数量合label数量不匹配"
            process_inputs.append(input_ids)
            process_labels.append(new_labels)
            
            
            #处理padding 这里策略是补齐到当前batch的最大长度
        finally_inputs_ids=[]
        finally_labels=[]
        max_len=max(len(seq) for seq in process_inputs)
        if max_len > 512:
            print(f"检测到超长样本，长度={max_len}，自动截断至512")
            process_inputs = [seq[:512] for seq in process_inputs]
            process_labels = [seq[:512] for seq in process_labels]
            max_len = 512

            
        for input_ids,labels in zip(process_inputs,process_labels):
            padding_length=max_len-len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length 
            labels += [-100] * padding_length

            finally_inputs_ids.append(input_ids)
            finally_labels.append(labels)
            
        batch_text = torch.tensor(finally_inputs_ids, dtype=torch.long)
        batch_label = torch.tensor(finally_labels, dtype=torch.long)

        return batch_text, batch_label





# base_dir = os.path.dirname(__file__)  # 获取当前脚本所在路径
# train_file_path = os.path.join(base_dir, "data", "msra_NER","train.txt")
# val_file_path= os.path.join(base_dir, "data", "msra_NER","test.txt")
# train_df,set_label= MyDataset.data_read(train_file_path)
# val_df,_= MyDataset.data_read(val_file_path)



# train_dataset = MyDataset(data=train_df,model_path="./model/models--hfl--chinese-bert-wwm/snapshots/ab0aa81da273504efc8540aa4d0bbaa3016a1bb5",file_path=train_file_path)
# val_dataset = MyDataset(data=val_df,model_path="./model/models--hfl--chinese-bert-wwm/snapshots/ab0aa81da273504efc8540aa4d0bbaa3016a1bb5",file_path=train_file_path)

# num_classes = len(MyDataset.idx_2_labels)

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,collate_fn=train_dataset.collate_fn_2)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,collate_fn=train_dataset.collate_fn_2)

# for batch_text,batch_label in train_loader:
#     print(train_dataset.tokenizer.decode(batch_text[1], skip_special_tokens=False))
#     print(batch_text[1])
#     print(batch_label[1])
#     print(batch_text.shape)
#     print(batch_label.shape)

