import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    自注意力机制模块，是Transformer功能模块的关键部分(注意力机制)
    作用：
    1.生成Q,K,V
    2.通过注意力机制来计算每一个位置对其他位置的关注权重
    3.通过多头机制，从多个角度来理解序列信息
    """

    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=embed_size//heads #每一个头的维度，这里采用的是先计算Q,K,V，在分成几个头
        """
        一般有两种方式，一种是先embedded后的数据*公用的WQ,WK,WV，然后得到Q,K,V，然后分成几个头，每一个头各自计算各自的attention
        另一种是将embedded后的数据分成head份，然后*每一个头各自不同的的WQ,WK,WV，然后得到各自的Q,K,V，然后计算每一个各自的attention
        第一种参数少，因为用的是公用的WQ,WK,WV，并且只计算一次Q,K,V,适合GPU的并行计算
        所以我们一般选择第一种
        """

        #确保词向量的维度能够整除
        assert(self.head_dim*heads==embed_size),"Embedding size 需要被整除"

        #用于计算Q,K,V的线性层（全连接层），每一层有embed_size个神经元，且每一个神经元有embed_size个w,这是为了保证输入输出的维度相等，且可以计算
        self.values=nn.Linear(embed_size,embed_size)
        self.queries=nn.Linear(embed_size,embed_size)
        self.keys=nn.Linear(embed_size,embed_size)

        #输出层，进行线性映射，将Q,K,V映射到原来的维度
        self.fc_out=nn.Linear(embed_size,embed_size)

    def forward(self,values,keys,query,mask):
        #计算Q,K,V
        values=self.values(values)
        keys=self.keys(keys)
        queries=self.queries(query)

        #样本数量
        N=values.size(0)
        #词的数量
        seq_len=values.size(1)

        #将Q,K,V矩阵分成heads份,所以每一份Q,K,V的维度(样本数量,词个数，每一个头的维度)
        values=values.reshape(N,seq_len,self.heads,self.head_dim)
        keys = keys.reshape(N, seq_len,self.heads, self.head_dim)
        queries = queries.reshape(N, seq_len, self.heads,self.head_dim)

        #计算注意力分数，每一个Q与每一个K，
        energy=torch.einsum("nqhd,nkhd->nhqk",[queries,keys])

        #padding mask，将填充位置的注意力分数设为极小值
        if mask is not None:
            #mask形状需要扩展到(batch_size, 1, seq_len, seq_len)以匹配energy的形状(batch_size, heads, seq_len, seq_len)
            #mask当前形状是(batch_size, 1, 1, seq_len)，需要扩展到(batch_size, 1, seq_len, seq_len)
            #mask表示哪些key位置是有效的，需要扩展到每个query位置
            mask = mask.expand(-1, -1, seq_len, -1)  # (batch_size, 1, seq_len, seq_len)
            energy=energy.masked_fill(mask==0,float("-1e20"))

        #将注意力分数转化为概率分布，即让关注度大的更大，让关注度小的更小
        attention_weight=torch.softmax(energy/(self.head_dim**0.5),dim=3)    #dim=3代表在最后一个维度进行softmax

        #用注意力对values进行加权求和，代表得到上下文的理解
        attention=torch.einsum("nhqk,nkhd->nqhd",[attention_weight,values])

        #将几个头拼接在一起
        out=attention.reshape(N, seq_len,self.heads*self.head_dim)

        #进行线性映射,返回以前的维度,输出维度(样本数量，词个数，词向量维度)
        out=self.fc_out(out)
        return out