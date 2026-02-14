import re

import jieba
from gensim.models import word2vec
import multiprocessing
from config import config
# 预训练词向量
def clean_text(text):
    # 移除 {%话题%}、[表情]、@用户名
    text = re.sub(r'{%.*?%}', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'@[\w\-]+', '', text)
    # 移除 emoji 表情
    emoji_pattern = re.compile("["
        u"\U0001F300-\U0001F6FF"  # 各类符号
        u"\U0001F700-\U0001F77F"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA70-\U0001FAFF"
        u"\u2600-\u26FF"  # 杂项符号
        u"\u200b"         # 零宽空格
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    # 替换中文全角标点为半角
    text = text.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
    # 去掉奇怪符号，比如《》《》――等
    text = re.sub(r'[《》“”"…—–~]', '', text)
    # 去掉多余空白
    text = re.sub(r'\s+', '', text)
    return text

def train_word2vec(data_path="./data/alldata.txt", output_path="./data/word2VecModel"):
    sentences = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            parts = line.split(',', 2)
            if len(parts) != 3:
                continue  # 不符合格式就跳过
            _, label, content = parts
            # temp = line.replace('\n', '').split(',,')
            temp = clean_text(content)
            if len(temp) >= 1:
                sentences.append(jieba.lcut(temp))

    model = word2vec.Word2Vec(
        sentences=sentences,
        vector_size=config.embeddingSize,
        min_count=config.miniFreq,
        window=10,
        workers=multiprocessing.cpu_count(),
        sg=1,
        epochs=20
    )
    model.save(output_path)
    print(f"Word2Vec model saved to {output_path}")


if __name__ == "__main__":
    train_word2vec()

