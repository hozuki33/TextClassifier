# TextClassifier
完成微博短文本情感分类。

## 核心特性
- 一站式流程：覆盖文本清洗、分词、特征提取、模型训练、评估、预测全流程；
- 可视化评估：自动生成混淆矩阵、准确率/召回率曲线等评估报告；
- 项目报告见文件夹中的word文档

## 环境依赖
```bash
# 基础依赖
Python >= 3.8
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
jieba >= 0.42.1  # 中文分词
nltk >= 3.7      # 英文分词/停用词

# 深度学习依赖
torch >= 1.9.0 或 tensorflow >= 2.6.0
gensim >= 4.1.0  # Word2Vec特征
