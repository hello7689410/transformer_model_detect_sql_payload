import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock

class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device

        #词嵌入层,将词表中的每一个词用embed_size维表示,创建一个查表矩阵
        self.word_embedding=nn.Embedding(vocab_size,embed_size)

        #位置编码,给序列（一句话）中每一个词的位置用embed_size维来表示
        self.position_embedding=nn.Embedding(max_length,embed_size)

        #堆叠Transformer功能模块,一共num_layers层Transformer
        self.layers=nn.ModuleList([
            TransformerBlock(
                embed_size,heads,forward_expansion,dropout
            )
            for _ in range(num_layers)
        ])

        #Dropout
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        """
        完整Encoder功能
        """
        #获取样本个数，每一个样本的序列长度
        N,seq_length=x.shape[0],x.shape[1]

        #生成位置索引
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)

        #词向量化
        word_embedding=self.word_embedding(x)

        #位置编码化
        position_embedding=self.position_embedding(positions)

        #词向量化+位置编码化,让模型不仅关注词的意思，该关注词的意思
        out=self.dropout(word_embedding+position_embedding)

        #将预处理好的数据放进Transformer模块
        for layer in self.layers:
            out=layer(out,out,out,mask)
        return out