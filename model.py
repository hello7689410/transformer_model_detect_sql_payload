import torch.nn as nn
from encoder import Encoder

class model(nn.Module):
    def __init__(self,vocab_size,pad_idx,embed_size=512,num_layers=6,forward_expansion=4,
                 heads=8,dropout=0,device='cuda',max_length=512,num_classes=1):
        super(model,self).__init__()

        #建立Encoder
        self.encoder=Encoder(vocab_size,embed_size,num_layers,heads,device,forward_expansion, dropout, max_length)

        #保存有效位置的掩码
        self.pad_idx=pad_idx
        
        #添加分类层，将encoder输出转换为分类结果
        self.fc=nn.Linear(embed_size,num_classes)

    def make_src_mask(self,src):
        """
        生成原序列的掩码，屏蔽填充位置，来让模型不关注这些位置
        """
        src_mask=(src!=self.pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def forward(self,x):
        #生成掩码
        src_mask=self.make_src_mask(x)

        #利用encoder来理解上下文
        out=self.encoder(x,src_mask)

        #对序列进行平均池化，得到句子级别的表示
        #out形状是(batch_size, seq_len, embed_size)
        #使用mask来忽略padding位置
        mask_expanded=src_mask.squeeze(1).squeeze(1).float()  # (batch_size, seq_len)
        mask_expanded=mask_expanded.unsqueeze(-1)  # (batch_size, seq_len, 1)
        out=out*mask_expanded  # 将padding位置置为0
        out=out.sum(dim=1)/mask_expanded.sum(dim=1)  # 平均池化，只计算有效位置
        
        #通过分类层得到分类结果
        out=self.fc(out)

        #返回分类结果
        return out



