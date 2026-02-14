import json
from collections import Counter
import numpy as np
import jieba
import gensim
from config import config
from preprocess import clean_text

# 数据预处理的类，生成训练集和测试集
class Dataset:
    def __init__(self, config):
        self.dataSource = config.dataSource
        self.stopWordSource = config.stopWordSource
        self.sequenceLength = config.sequenceLength
        self.embeddingSize = config.embeddingSize
        self.rate = config.rate
        self.miniFreq = config.miniFreq

        self.stopWordDict = {}
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []
        self.wordEmbedding = None
        self.n_symbols = 0
        self.wordToIndex = {}
        self.indexToWord = {}

    def readData(self, filePath):
        text, label = [], []
        with open(filePath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(',', 2)  # 最多分 3 个字段：ID、标签、正文
                if len(parts) != 3:
                    continue
                _, lab, content = parts  # 跳过 ID，只取标签和内容
                content = clean_text(content)
                text.append(content)
                label.append(int(lab))
                # temp = line.replace('\n', '').split(',,')
                # if len(temp) >= 2:
                #     text.append(temp[0])
                #     label.append(temp[1])
        texts = [jieba.lcut(document.replace('\n', '')) for document in text]
        return texts, label

    def readStopWord(self, stopWordPath):
        # with open(stopWordPath, "r", encoding="utf-8") as f:
        #     stopWords = f.read().splitlines()
        #     self.stopWordDict = dict(zip(stopWords, range(len(stopWords))))
        with open(stopWordPath, "r", encoding="utf-8") as f:
            stopWords = [w.strip() for w in f if w.strip()]
            self.stopWordDict = dict.fromkeys(stopWords)

    def getWordEmbedding(self, words):
        model = gensim.models.Word2Vec.load('./data/word2VecModel')
        vocab = ["pad", "UNK"]
        wordEmbedding = [np.zeros(self.embeddingSize), np.random.randn(self.embeddingSize)]
        missed = 0

        for word in words:
            if word in model.wv:
                vocab.append(word)
                wordEmbedding.append(model.wv[word])
            else:
                missed += 1
                if missed <= 5:
                    print(f"{word} 不存在于词向量中")
        if missed > 5:
            print(f"...共 {missed} 个词不在词向量中。")
        return vocab, np.array(wordEmbedding)

    def genVocabulary(self, reviews):
        allWords = [word for review in reviews for word in review]
        subWords = [word for word in allWords if word not in self.stopWordDict]
        wordCount = Counter(subWords)
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sortWordCount if item[1] >= self.miniFreq]

        vocab, wordEmbedding = self.getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        self.wordToIndex = dict(zip(vocab, range(len(vocab))))
        self.indexToWord = dict(zip(range(len(vocab)), vocab))
        self.n_symbols = len(self.wordToIndex)

        with open("data/wordJson/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self.wordToIndex, f)
        with open("data/wordJson/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self.indexToWord, f)

    def reviewProcess(self, review):
        # reviewVec = np.zeros(self.sequenceLength)
        reviewVec = np.zeros(self.sequenceLength, dtype=np.int64)
        sequenceLen = min(len(review), self.sequenceLength)
        for i in range(sequenceLen):
            reviewVec[i] = self.wordToIndex.get(review[i], self.wordToIndex["UNK"])
        return reviewVec

    def genTrainEvalData(self, x, y):
        reviews, labels = [], []
        for i in range(len(x)):
            reviews.append(self.reviewProcess(x[i]))
            labels.append([int(y[i])])
        trainIndex = int(len(x) * self.rate)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")
        return trainReviews, trainLabels, evalReviews, evalLabels

    def dataGen(self):
        self.readStopWord(self.stopWordSource)
        reviews, labels = self.readData(self.dataSource)
        self.genVocabulary(reviews)
        self.trainReviews, self.trainLabels, self.evalReviews, self.evalLabels = self.genTrainEvalData(reviews, labels)


if __name__ == "__main__":
    data = Dataset(config)
    data.dataGen()
    print("Train Data Shape:", data.trainReviews.shape)
    print("Train Label Shape:", data.trainLabels.shape)
    print("Eval Data Shape:", data.evalReviews.shape)
