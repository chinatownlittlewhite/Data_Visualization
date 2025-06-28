import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import os
from datetime import datetime

# --- 配置参数 ---
MODEL_NAME = 'distilbert-base-uncased'  # 使用的预训练模型名称
OUTPUT_DIR = './distilbert-imdb-finetuned'  # 输出目录
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_imdb_data.csv')  # 处理后的数据保存路径
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'distilbert-imdb-finetuned')  # 模型保存路径
MAX_REVIEWS_TO_PROCESS = 50000  # 最大处理的评论数量，设置为50000可使用完整数据集

# --- 设备设置 ---
# 自动检测可用设备，优先级：CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DEVICE_NAME = 'CUDA'
elif torch.backends.mps.is_available():  # 检查MPS(Metal Performance Shaders)是否可用(苹果芯片)
    DEVICE = torch.device('mps')
    DEVICE_NAME = 'MPS'
else:
    DEVICE = torch.device('cpu')
    DEVICE_NAME = 'CPU'

print(f"使用设备: {DEVICE_NAME}")


# --- 辅助函数 ---
def clean_text(text):
    """清理文本：移除HTML标签和多余空格"""
    text = re.sub(r'<br\s*/?>', ' ', text)  # 移除<br>标签
    text = re.sub(r'\s+', ' ', text).strip()  # 合并多个空格为一个
    return text


def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids  # 真实标签
    preds = pred.predictions.argmax(-1)  # 预测结果
    # 计算精确率、召回率、F1值（二分类）
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)  # 计算准确率
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# --- PyTorch数据集类 ---
class IMDbDataset(torch.utils.data.Dataset):
    """自定义IMDb评论数据集类"""

    def __init__(self, encodings, labels):
        self.encodings = encodings  # 编码后的文本
        self.labels = labels  # 标签

    def __getitem__(self, idx):
        # 获取单个样本
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)  # 数据集大小


# --- 主程序逻辑 ---
def main():
    print("--- 开始IMDb情感分析训练 ---")
    print(f"使用设备: {DEVICE_NAME}")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 数据加载和处理 ---
    print("--- 步骤1: 加载和合并数据 ---")

    try:
        # 加载IMDb评论数据
        reviews_df = pd.read_csv('./data/IMDB Dataset.csv')
        reviews_df = reviews_df.head(MAX_REVIEWS_TO_PROCESS)  # 使用子集

        # 加载IMDb元数据
        print("加载title.basics...")
        basics_df = pd.read_csv('./data/title.basics.tsv.gz', sep='\t', low_memory=False,
                                usecols=['tconst', 'titleType', 'primaryTitle', 'startYear', 'genres'])
        print("加载title.ratings...")
        ratings_df = pd.read_csv('./data/title.ratings.tsv.gz', sep='\t', low_memory=False)

        # 筛选电影数据
        movies_df = basics_df[basics_df['titleType'] == 'movie']

        # 合并基本信息和评分
        metadata_df = pd.merge(movies_df, ratings_df, on='tconst')

        # 数据清理
        metadata_df = metadata_df.dropna(subset=['startYear', 'genres', 'averageRating'])
        metadata_df = metadata_df[metadata_df['genres'] != '\\N']
        metadata_df['startYear'] = pd.to_numeric(metadata_df['startYear'], errors='coerce')
        metadata_df = metadata_df.dropna(subset=['startYear'])
        metadata_df['startYear'] = metadata_df['startYear'].astype(int)

        # 随机采样元数据以匹配评论数量
        sampled_metadata = metadata_df.sample(n=len(reviews_df), replace=True).reset_index(drop=True)

        # 合并评论和元数据
        reviews_df['primaryTitle'] = sampled_metadata['primaryTitle']
        reviews_df['startYear'] = sampled_metadata['startYear']
        reviews_df['genres'] = sampled_metadata['genres']
        reviews_df['averageRating'] = sampled_metadata['averageRating']
        reviews_df['numVotes'] = sampled_metadata['numVotes']

        print("--- 步骤2: 数据清理和预处理 ---")
        reviews_df['review'] = reviews_df['review'].apply(clean_text)  # 清理文本
        reviews_df['sentiment'] = reviews_df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # 转换情感标签

        # 保存处理后的数据供Streamlit应用使用
        reviews_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"处理后的数据已保存到 {PROCESSED_DATA_PATH}")

        # --- 模型训练 ---
        print("--- 步骤3: 模型微调 ---")

        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            reviews_df['review'].tolist(),
            reviews_df['sentiment'].tolist(),
            test_size=0.2,
            random_state=42
        )

        # 初始化tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        # 对文本进行编码
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        # 创建数据集
        train_dataset = IMDbDataset(train_encodings, train_labels)
        val_dataset = IMDbDataset(val_encodings, val_labels)

        # 加载模型并移动到指定设备
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(DEVICE)

        # 定义训练参数
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_results_dir = os.path.join(OUTPUT_DIR, f'training_results_{timestamp}')

        training_args = TrainingArguments(
            output_dir=training_results_dir,  # 输出目录
            num_train_epochs=10,  # 训练轮数
            per_device_train_batch_size=16,  # 训练批次大小
            per_device_eval_batch_size=16,  # 评估批次大小
            warmup_steps=500,  # 预热步数
            weight_decay=0.01,  # 权重衰减
            logging_dir=os.path.join(training_results_dir, 'logs'),  # 日志目录
            logging_steps=100,  # 日志记录间隔
            evaluation_strategy="epoch",  # 每个epoch后评估
            save_strategy="epoch",  # 每个epoch后保存
            load_best_model_at_end=True,  # 训练结束时加载最佳模型
            report_to="tensorboard",  # 使用TensorBoard记录
            save_total_limit=2,  # 最多保存2个检查点
            fp16=DEVICE_NAME == 'CUDA',  # CUDA设备启用混合精度训练
            seed=42,  # 随机种子
            run_name=f"distilbert-imdb-{timestamp}",  # 运行名称
        )

        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # 开始训练
        print("开始训练...")
        trainer.train()
        print("训练完成.")

        # 保存微调后的模型和tokenizer
        trainer.save_model(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print(f"模型已保存到 {MODEL_SAVE_PATH}")

        # 保存训练指标
        training_metrics = trainer.evaluate()
        with open(os.path.join(training_results_dir, 'final_metrics.txt'), 'w') as f:
            for key, value in training_metrics.items():
                f.write(f"{key}: {value}\n")

    except Exception as e:
        print(f"发生错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()