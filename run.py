import torch.nn as nn
import torch
from model import model #导入模型，生成上下文的理解
from bbpe import load_sql_injection_data, vectorize_with_bbpe
from torch.utils.data import TensorDataset,DataLoader
import predict
"""
load_sql_injection_data:读取数据和标签
vectorize_with_bbpe:文本向量化，即将每一个文本转化为token ID序列
"""

#处理和读取数据
def load_data():
    # 读取正常数据和标签
    normal_csv = r"data\train\白.csv"
    normal_texts, normal_labels = load_sql_injection_data(normal_csv, text_col=3, label_col=6)
    print("正常数据：",len(normal_texts),len(normal_labels))
    # 读取恶意文本（SQL注入.csv)
    sql_csv = r"data\train\SQL注入.csv"
    sql_texts, sql_labels = load_sql_injection_data(sql_csv, text_col=3, label_col=6)
    print("SQL数据：",len(sql_texts),len(sql_labels))
    # 合并正常数据和SQL注入数据
    texts = normal_texts + sql_texts
    labels = normal_labels + sql_labels
    print("样本总数:",len(labels),"1的总数:",labels.count(1),"0的总数:",labels.count(0))
    x,tokenizer=vectorize_with_bbpe(texts, model_path="tokenizer.json", num_merges=1000, sequences_path="train_sequences.json")
    x=x[:,:512].long()
    #将标签的类型化为张量
    y=torch.tensor(labels,dtype=torch.long)

    print(len(x),len(y))
    #将数据和标签封装在Dataset
    dataset=TensorDataset(x,y)
    #将数据封装遭Dataloader
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True)

    #返回这个迭代器(这样每一次训练可以取32份数据出来)
    return dataloader,len(tokenizer.token2id),tokenizer

#一轮训练
def train(dataloader,model,optimizer,criterion):
    device='cuda' if torch.cuda.is_available() else "cpu"
    #按批次训练，每一个批次取batch_size=32个样本进行训练
    for batch_idx,(data,labels) in enumerate(dataloader):
        data=data.to(device)
        labels=labels.to(device).float().unsqueeze(1)
        y_pred=model(data)  #正向传播
        loss=criterion(y_pred,labels)   #计算损失
        optimizer.zero_grad()           #梯度置零，因为梯度会相加
        loss.backward()                 #计算梯度
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                #利用梯度来更新参数
        print(loss.item())

#人为设置权重
def get_pos_weight():
    #统计正常与异常的比例
    total_pos = 0
    total_neg = 0
    for batch_data, batch_labels in dataloader:
        batch_pos = (batch_labels == 1).sum().item()
        batch_neg = (batch_labels == 0).sum().item()
        total_pos += batch_pos
        total_neg += batch_neg
    #设置权重,给异常数据设置权重:0的数量/1的数量
    pos_weight_value = total_neg / total_pos
    print(f"[Loss] total_pos={total_pos}, total_neg={total_neg}, pos_weight={pos_weight_value:.4f}")
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return pos_weight_tensor


if __name__=="__main__":
    #读取和处理数据
    dataloader,vocab_size,tokenizer=load_data()

    #设置模型参数
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_idx = 0
    embed_size = 512
    num_layers = 6
    forward_expansion = 4
    heads = 8
    dropout = 0.1
    max_length = 512
    num_classes = 1   # 二分类，用 BCEWithLogitsLoss，输出为 [0,1]

    # 创建模型
    model=model(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        embed_size=embed_size,
        num_layers=num_layers,
        forward_expansion=forward_expansion,
        heads=heads,
        dropout=dropout,
        device=device,
        max_length=max_length,
        num_classes=num_classes
    )
    model.to(device)
    model.train()


    pos_weight=get_pos_weight()
    #设置损失函数
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #定义优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)

    #开始训练：正向传播与反向传播.一个运行50轮。每一轮的每一次训练取32个样本
    for epoch in range(20):
        if epoch==0:
            print("开始训练")
        train(dataloader,model,optimizer,criterion)
        print(f"第{epoch}次训练结束")

    #保存模型
    checkpoint = {
        "model_state_dict": model.state_dict(),  # 模型参数
        # 下面这些超参数用于在 predict.py 中正确重建模型结构
        "vocab_size": int(vocab_size),
        "pad_idx": int(pad_idx),
        "embed_size": int(embed_size),
        "num_layers": int(num_layers),
        "forward_expansion": int(forward_expansion),
        "heads": int(heads),
        "dropout": float(dropout),
        "max_length": int(max_length),
        "num_classes": int(num_classes),
    }

    torch.save(checkpoint, "sql_classifier_model.pth")
    print("训练完成，模型已保存为: sql_classifier_model.pth")

    print("开始进行预测")
    predict.main("test")
    predict.main("train")

