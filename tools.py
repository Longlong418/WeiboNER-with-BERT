
#获取实体
def extract_entities(labels)->list: #return [(seq_idx,entity_type,start_idx,end_idx),...]
    """
    labels的格式为：
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