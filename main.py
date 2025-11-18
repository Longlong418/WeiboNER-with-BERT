from data_process import MyDataset
from model import BertCnnNER,BertNER
from My_Config import My_Config
import argparse
from train_evaluate import train,evaluate
from tools import predict_sentence
import torch
from datetime import datetime
import os
def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="NER")
    parser.add_argument("--config_path", type=str, default="./NER_Config/Bertbase_Weibo_Config.json")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "predict"], required=True)
    args = parser.parse_args()
    #参数配置
    config=My_Config(config_path=args.config_path)
    #初始化 MyDataset.idx_2_labels
    _, set_label = MyDataset.data_read(config.train_path)
    MyDataset.idx_2_labels = sorted(set_label)
    MyDataset.labels_2_idx = {j: i for i, j in enumerate(MyDataset.idx_2_labels)}
    num_classes = len(MyDataset.idx_2_labels)
    model = BertCnnNER(num_classes=num_classes,config=config).to(config.device)

    if args.mode == "train":
        train(model, config)
    elif args.mode == "eval":
        if not config.trained_model_path:
            raise ValueError("请先进行模型训练!")
        model.load_state_dict(torch.load(config.trained_model_path))
        val_file_path=config.test_path
        val_f1,val_precision,val_recall=evaluate(model, config,val_file_path)
        print(f'Val F1: {val_f1 :.4f},\
            Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        #保存指标到training_log.txt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        result_path = os.path.join(config.trained_save_root_path, "training_log.txt")
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f"模型：{config.model_name} 数据集: {config.data_name}\n")
            f.write(f"[{timestamp}]\nVal F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n\n")

    elif args.mode =='predict':
        if not config.trained_model_path:
            raise ValueError("请先进行模型训练!")
        model.load_state_dict(torch.load(config.trained_model_path))
        text=input("请输入你要预测的语句:")
        print(predict_sentence(model=model,text=text,config=config))

if __name__=="__main__":
    main()


    