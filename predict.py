import json
import re
from urllib.parse import quote

import torch
import os
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
#TensorDataset:数据集  DataLoader:加载器，即给数据集加一个迭代的功能

from model import model as Model
from bbpe import load_sql_injection_data,vectorize_with_bbpe

def load_data(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取数据和标签 —— 统一使用 load_sql_injection_data，保证标签安全转为 0/1
    # 读取正常文本（白.csv）
    normal_csv = rf"data\{path}\白.csv"
    print(f"\n加载正常文本: {normal_csv}")
    normal_texts, normal_labels = load_sql_injection_data(normal_csv, text_col=3, label_col=6)

    # 读取恶意文本（SQL注入.csv）
    sql_csv = rf"data\{path}\SQL注入.csv"
    print(f"\n加载恶意文本: {sql_csv}")
    sql_texts, sql_labels = load_sql_injection_data(sql_csv, text_col=3, label_col=6)

    print(type(sql_texts))
    print(type(normal_texts))
    # 合并恶意和正常
    text = normal_texts + sql_texts  # text 列里存放的就是每一条 URL / 请求文本
    labels = normal_labels + sql_labels
    print("1的数量", labels.count(1), "0的数量", labels.count(0))

    # 进行编码化，即合并字，并且给每一个子词分配索引
    x, tokenizer = vectorize_with_bbpe(
        text,
        model_path="tokenizer.json",
        num_merges=1000,
        sequences_path=f"{path}_sequences.json",
    )
    x = x[:, :512].long().to(device)

    dataset = TensorDataset(x)
    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    # 现在多返回 text，后续用于打印预测失败时对应的 URL / 文本
    return loader, labels, text

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


def print_misclassified_samples(texts, labels, preds, path: str = ""):
    """
    打印预测失败的样本及其对应的 URL / 文本。

    :param texts: 原始文本列表
    :param labels: 真实标签列表（0/1）
    :param preds: 模型预测结果（Tensor 或 list）
    :param path: 数据集名称（如 'train' / 'test'），仅用于打印时标识
    """
    # 将 preds 统一转成 python list，方便遍历比较
    if hasattr(preds, "cpu"):
        preds_list = preds.cpu().tolist()
    else:
        preds_list = list(preds)

    print(f"\n=== [{path}] 预测错误样本（URL / 文本）列表 ===")
    error_count = 0

    for idx, (t, y_true, y_pred) in enumerate(zip(texts, labels, preds_list)):
        if int(y_true) != int(y_pred):
            error_count += 1
            # 只截取前面一部分，避免太长占满终端
            preview = t if len(str(t)) <= 300 else str(t)[:300] + "..."
            print(f"[样本 {idx}] 真实={y_true}, 预测={y_pred}, 文本/URL: {preview}")

    if error_count == 0:
        print(f"[{path}] 本次没有预测错误的样本 ✅")
    else:
        print(f"[{path}] 共 {error_count} 条样本预测错误 ❌")

def _normalize_for_model(text: str) -> str:
    """
    统一做一次简单清洗 + URL 编码，作为送入模型的最终文本。
    这样所有调用方都可以直接传入“原始字符串”（URL、payload 等）。
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    return quote(raw, safe="")


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

def main(path):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #读取和处理数据
    loader, labels, texts = load_data(path)

    #加载模型
    classifier_model,  ckpt_vocab_size=load_model()

    #正向传播
    all_preds = forward(loader, classifier_model, ckpt_vocab_size)

    # 打印预测失败的 URL / 文本
    print_misclassified_samples(texts, labels, all_preds, path=path)

    #保存结果
    save(path, all_preds, labels)
if __name__=="__main__":
    main("train")