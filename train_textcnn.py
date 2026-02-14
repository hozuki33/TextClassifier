import numpy as np
import yaml
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, concatenate, Flatten, Dense, Dropout, Embedding, Reshape
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers, optimizers, losses
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os  # 新增：用于目录操作

from dataset import Dataset
from config import config  # 确保你有 config.py 并设置好超参数


# 定义卷积模块
def convolution(config):
    sequence_length = config.sequenceLength
    embedding_dimension = config.embeddingSize
    inn = Input(shape=(sequence_length, embedding_dimension, 1))
    cnns = []
    for size in config.filterSizes:
        conv = Conv2D(filters=config.numFilters, kernel_size=(size, embedding_dimension),
                      strides=1, padding='valid', activation='relu')(inn)
        pool = MaxPool2D(pool_size=(sequence_length - size + 1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt = concatenate(cnns)
    return Model(inputs=inn, outputs=outt)


# 定义完整模型
def cnn_mulfilter(n_symbols, embedding_weights, config):
    # 确保词向量矩阵的形状正确
    if embedding_weights.shape[0] != n_symbols:
        print(f"警告: 词向量矩阵形状 {embedding_weights.shape} 与词表大小 {n_symbols} 不匹配")
        print(f"将调整词向量矩阵以匹配词表大小")

        # 创建新的词向量矩阵，包含随机初始化的未知词
        new_embedding = np.random.normal(
            size=(n_symbols, config.embeddingSize)
        )
        # 复制预训练向量中存在的词
        min_length = min(n_symbols, embedding_weights.shape[0])
        new_embedding[:min_length] = embedding_weights[:min_length]
        embedding_weights = new_embedding

    model = Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                  weights=[embedding_weights], input_length=config.sequenceLength),
        Reshape((config.sequenceLength, config.embeddingSize, 1)),
        convolution(config),
        Flatten(),
        Dense(10, activation='relu', kernel_regularizer=regularizers.l2(config.l2RegLambda)),
        Dropout(config.dropoutKeepProb),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


# 确保模型保存目录存在
model_dir = './model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, 'best_model'))
    print(f"已创建模型保存目录: {model_dir}")

# 准备数据
data = Dataset(config)
data.dataGen()
x_train, y_train = data.trainReviews, data.trainLabels
x_eval, y_eval = data.evalReviews, data.evalLabels
wordEmbedding = data.wordEmbedding

# 修正词表大小：使用词向量矩阵的行数
n_symbols = wordEmbedding.shape[0]  # 应该是39788

# 打印调试信息
print(f"修正后的词表大小: {n_symbols}")
print(f"词向量矩阵形状: {wordEmbedding.shape}")

# 初始化模型
model = cnn_mulfilter(n_symbols, wordEmbedding, config)
model.summary()

# 回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5'),
    save_best_only=True, save_weights_only=True, verbose=1
)

# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=config.batchSize,
    epochs=config.epochs,
    validation_split=0.3,
    shuffle=True,
    callbacks=[reduce_lr, early_stopping, model_checkpoint]
)

# 评估模型
scores = model.evaluate(x_eval, y_eval)
print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

# 保存模型结构与权重（修改为JSON格式）
try:
    # 保存模型结构为JSON
    json_string = model.to_json()
    json_path = os.path.join(model_dir, 'textCNN.json')
    with open(json_path, 'w') as outfile:
        outfile.write(json_string)
    print(f"模型结构已保存至: {json_path}")

    # 保存模型权重
    weights_path = os.path.join(model_dir, 'textCNN.h5')
    model.save_weights(weights_path)
    print(f"模型权重已保存至: {weights_path}")

except Exception as e:
    print(f"保存模型时发生错误: {e}")
    print("仅保存模型权重作为备选方案")
    weights_path = os.path.join(model_dir, 'textCNN_fallback.h5')
    model.save_weights(weights_path)
    print(f"模型权重已保存至: {weights_path}")

import matplotlib.pyplot as plt

# 训练过程中的 accuracy 和 loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 绘制 Loss 曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
