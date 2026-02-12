from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import time
import json
import os

import pandas as pd
import torch
"""
Counter:统计字符对频率
defaultdict:建立词表
typing:类型标注
time:记时
json:用来保存tokenizer
pandas:读取CSV
torch:构建张量
"""

class BBPE:
    """
    分词器：
    原理：设置一个阈值，当两个字母或者单词经常一起出现的，并且要挨在一起的，当这个频率超过设置的阈值时，就会组合成
    新的单词。例如:se，lect，这两个词同时出现的频率超过阈值，那么就会组成新的词
    """

    def __init__(self, num_merges: int = 1000, min_freq: int = 2):
        """
        num_merges:最大合并次数，决定合成的词的长度
        min_freq:决定一个字符对要不要合并，即设置的阈值，当两个字符对同时出现且挨在一起，那么
        他们才会组成新的单词
        "merges":记录合并规则
        token2id:给合并完成后的所有词设置一个索引：给模型看的，用于embedding化
        id2token:把模型输出的数字转化为词表对应索引的字符对：给人看的，将模型输出的文字转化为数字
        """
        self.num_merges = num_merges
        self.min_freq = min_freq
        self.merges: Dict[Tuple[str, str], int] = {}
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

    def save(self, path: str) -> None:
        """
        保存功能，将生成好的tokenizer保存在json中，其实也是合并规则
        """
        #将合并规则保存在merges这个字典中：key->(字符对a，字符对b) 值->在词表对应的索引
        merges_list = [
            {"a": a, "b": b, "rank": rank} for (a, b), rank in self.merges.items()
        ]

        obj = {
            "version": 1,
            "num_merges": self.num_merges,
            "min_freq": self.min_freq,
            "merges": merges_list,
            "token2id": self.token2id,
        }
        """
        构造tokenizer配置：
        version:文本格式版本
        num_merges:最大合并次数
        min_freq:阈值：频率超过这个值，才能够合并
        merges:合并规则
        token2id:词表，token->id  
        """
        #写入文件中
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BBPE":
        """
        加载bbpe
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        #创建分词器类：最大合并次数1000,频率为20
        tokenizer = cls(
            num_merges=int(obj.get("num_merges", 1000)),
            min_freq=int(obj.get("min_freq", 20)),
        )

        #获取合并规则
        merges = obj.get("merges", [])
        tokenizer.merges = {(m["a"], m["b"]): int(m["rank"]) for m in merges}
        #获取token->id的词表，在obj里面的词表提取出来
        tokenizer.token2id = {k: int(v) for k, v in obj.get("token2id", {}).items()}
        #获取id->token的词表，直接利用token->id来构造出来
        tokenizer.id2token = {i: t for t, i in tokenizer.token2id.items()}
        return tokenizer    #返回这个词表，即下一次可以使用这个功能直接获取json里面的tokenizer

    @staticmethod
    def _word_to_symbols(word: str) -> Tuple[str, ...]:
       #将字符串中的每一个字符提取出来，然后再末尾加一个/w，
       # 例如,hello，返回['h','e','l','l','o','/w']
        return tuple(list(word) + ["</w>"])

    #统计相邻字符对出现的频率
    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:

        pairs = Counter()  # 用于统计字符对频率

        # 遍历词汇表中的每个词
        for word, freq in vocab.items():
            # 如果词的长度小于2，无法形成字符对，跳过
            if len(word) < 2:
                continue
            
            # 统计该词中所有相邻字符对
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])  #得到相邻字符对
                # 累加频率：如果这个词出现了 freq 次，那么这个字符对也出现了 freq 次
                pairs[pair] += freq     #将这些统计好的字符对放进pairs里面，假如这个字符对的频率超过某一个
                #阈值，就能在pairs这个字典中形成这个字符对,所以word[i]可能等于sel这个，这样就能够形成很长的字符对
        return pairs        #返回目前的字典里面所有相邻的字符对

    @staticmethod
    #执行一次合并规则
    def _merge_vocab(
        pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """
        pair:要合并的字符对
        vocab:当前词汇表
        """

        new_vocab: Dict[Tuple[str, ...], int] = {}  # 新的词汇表：字符对->索引
        bigram = pair  # 字符对（二元组）
        first, second = bigram  # 拆分成两个字符
        new_symbol = first + second  # 合并后的新符号

        # 遍历词汇表中的每个词
        for word, freq in vocab.items():
            word_list = list(word)  # 转换为列表以便修改
            i = 0
            new_word: List[str] = []  # 存储合并后的新词

            # 从左到右扫描词，查找并合并指定的字符对
            while i < len(word_list):
                # 如果当前位置和下一个位置正好是要合并的字符对
                if (
                    i < len(word_list) - 1  # 确保有下一个字符
                    and word_list[i] == first
                    and word_list[i + 1] == second
                ):
                    # 合并这两个字符
                    new_word.append(new_symbol)
                    i += 2  # 跳过已合并的两个字符
                else:
                    # 不是要合并的字符对，保留原字符
                    new_word.append(word_list[i])
                    i += 1

            # 将合并后的词（转换为元组）添加到新词汇表，保持原频率
            new_vocab[tuple(new_word)] = freq

        return new_vocab

    def fit(self, texts: List[str], verbose: bool = True,max_token_length=8) -> None:
        """
        在一组文本上训练 BBPE。
        
        Args:
            texts: 训练文本列表，即从文件里面读取的所有的单词
            verbose: 是否显示训练进度
        """
        print("开始构造tokenizer")
        vocab: Dict[Tuple[str, ...], int] = defaultdict(int)    #构建词表(哈希表)，{('s','l'),10}这种
        #键文件里面每一行的每一个单词，拆成字母，放进vocab这个词表里面，并且统计这个频率
        for line in texts:
            for word in line.strip().split():
                vocab[self._word_to_symbols(word)] += 1
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

        # 建立最终 token vocab
        if verbose:
            print("[BBPE] 步骤3/3: 构建最终 token 映射...")
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

    def encode(self, text: str) -> List[int]:
        """
        把一行文本编码成 token id 序列。
        """
        token_ids: List[int] = []
        for i,word in enumerate(text.strip().split()):   #遍历每一个词，并把词化为单个字母
            pieces = self.encode_word(word) #按照merges合并
            for p in pieces:                #变量完成合并后的词的每一个值，比如['s','le','c','t']
                if p in self.token2id:      #将每一个值完成映射，在token2id上面
                    token_ids.append(self.token2id[p])
        return token_ids                    #返回这句话的每一个词的子词，然后再token2id上完成映射的过程

    def encode_batch(self, texts: List[str]) -> List[List[int]]:    #对所有样本
        return [self.encode(t) for t in texts]


def vectorize_with_bbpe(
    texts,
    num_merges: int = 1000,     #合并次数
    min_freq: int = 2,          #阈值
    verbose: bool = True,       #用于是否打印
    model_path="tokenizer.json",   #路径，如果有这个词表（BBPE），就加载出来
    sequences_path: Optional[str] = None,  #用于保存编码后的sequences，默认路径为1_sequences.json
):
    #判断是否训练好词表，如果训练好，直接load这个词表，如果没有训练好，那么fit
    if model_path and os.path.exists(model_path):
        print("tokenzier已经存在，不需要在生成")
        tokenizer = BBPE.load(model_path)
    else:
        tokenizer = BBPE(num_merges=num_merges, min_freq=min_freq)
        tokenizer.fit(texts)
        tokenizer.save(model_path)
        print(f"Tokenizer 已保存为: {model_path}")

    #先检查是否有保存的sequences文件，如果有就直接加载，否则进行编码
    padded = None
    if sequences_path and os.path.exists(sequences_path):
        print("存在编码好的文件，直接加载文件")
        with open(sequences_path, 'r', encoding='utf-8') as f:
            padded = json.load(f)
    if padded is None:
        print("未存在有效的编码缓存，开始进行编码，对文件开始进行编码")
        sequences = tokenizer.encode_batch(texts)   #把文本中的所有词利用merges和id2token转化为对应的索引
        #将每一个序列的长度都化为max_len
        max_len = 512  # 设置为512，保留更多特征
        sequences = [seq[:max_len] for seq in sequences]

        # padding 到同一长度，方便后续喂给 Transformer
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(seq + [0] * pad_len)

        #保存padded(编码后的结果)为文件
        if sequences_path:
            with open(sequences_path, 'w', encoding='utf-8') as f:
                # padded 是个 list of lists，可以直接转为 json
                json.dump(padded, f)

    return torch.tensor(padded, dtype=torch.long), tokenizer


def load_sql_injection_data(csv_path: str, text_col: int = 3, label_col: int = 6, encoding: str = None):
    """
   第4列为sql payload
   第6列为标签
    """
    # 尝试不同的编码方式
    if encoding is None:
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(csv_path, header=0, low_memory=False, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        if df is None:
            raise ValueError(f"无法使用任何编码读取文件: {csv_path}")
    else:
        df = pd.read_csv(csv_path, header=0, low_memory=False, encoding=encoding)

    # 提取文本列（URL列）
    texts = df.iloc[:, text_col].astype(str).tolist()

    # 提取标签列，更健壮地处理各种情况
    label_series = df.iloc[:, label_col]

    # 对标签进行处理的函数，1化为数字1，其他的都为0
    def safe_to_int(x):
        try:
            val = float(x)
            if pd.isna(val):
                return 0
            return int(val)
        except (ValueError, TypeError):
            return 0

    labels = label_series.apply(safe_to_int).tolist()  # 将函数应用到labels_series里面，并且转化为列表
    return texts, labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
   # 1. 读取 SQL 注入数据
    normal_csv = rf"D:\桌面\机器学习\识别SQL-Transformer\data\train\白.csv"
    print(f"\n加载正常文本: {normal_csv}")
    texts, normal_labels = load_sql_injection_data(normal_csv, text_col=3, label_col=6)
    texts=[texts[0]]
    print(texts)
    x,t=vectorize_with_bbpe(texts,sequences_path="1.json")
    print(x)