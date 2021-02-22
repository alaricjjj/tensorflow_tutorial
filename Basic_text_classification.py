# 我们将使用来源于网络电影数据库（Internet Movie Database）的 IMDB 数据集（IMDB dataset），
# 其包含 50,000 条影评文本。从该数据集切割出的25,000条评论用作训练，另外 25,000 条用作测试。
# 训练集与测试集是平衡的（balanced），意味着它们包含相等数量的积极和消极评论。

# 此笔记本（notebook）使用了 tf.keras，它是一个 Tensorflow 中用于构建和训练模型的高级API。
# 有关使用 tf.keras 进行文本分类的更高级教程，请参阅 MLCC文本分类指南（MLCC Text Classification Guide）。

import tensorflow as tf
from tensorflow import keras

import numpy as np

# print(tf.__version__)

'''下载 IMDB 数据集'''
# IMDB 数据集已经打包在 Tensorflow 中。该数据集已经经过预处理，评论（单词序列）已经被转换为整数序列，其中每个整数表示字典中的特定单词。
# 以下代码将下载 IMDB 数据集到您的机器上（如果您已经下载过将从缓存中复制）：
imdb = keras.datasets.imdb

'''探索数据'''
# data = imdb.load_data(num_words=10000)
# print(type(data))
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# print(train_data[0])

# 电影评论可能具有不同的长度。以下代码显示了第一条和第二条评论的中单词数量。由于神经网络的输入必须是统一的长度，我们稍后需要解决这个问题。
# print(len(train_data[0]), len(train_data[1]))

# 将整数转换回单词
# 了解如何将整数转换回文本对您可能是有帮助的。这里我们将创建一个辅助函数来查询一个包含了整数到字符串映射的字典对象：

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 现在我们可以使用 decode_review 函数来显示首条评论的文本：
print('Train data 0 info', decode_review(train_data[0]))

'准备数据'
# 影评——即整数数组必须在输入神经网络之前转换为张量。这种转换可以通过以下两种方式来完成：
#
    # 将数组转换为表示单词出现与否的由 0 和 1 组成的向量，类似于 one-hot 编码。例如，序列[3, 5]
    # 将转换为一个 10,000 维的向量，该向量除了索引为 3 和 5 的位置是 1 以外，其他都为 0。然后，
    # 将其作为网络的首层——一个可以处理浮点型向量数据的稠密层。不过，这种方法需要大量的内存，需要一
    # 个大小为 num_words * num_reviews 的矩阵。
    #
    # 或者，我们可以填充数组来保证输入数据具有相同的长度，然后创建一个大小为
    # max_length * num_reviews 的整型张量。我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层。

# 在本教程中，我们将使用第二种方法。
#
# 由于电影评论长度必须相同，我们将使用 pad_sequences 函数来使长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 现在让我们看下样本的长度：
print(len(train_data[0]), len(train_data[1]))

# 并检查一下首条评论（当前已经填充）：
print(train_data[0])

'''构建模型'''
# 神经网络由堆叠的层来构建，这需要从两个主要方面来进行体系结构决策：

# 模型里有多少层？
# 每个层里有多少隐层单元（hidden units）？
# 在此样本中，输入数据包含一个单词索引的数组。要预测的标签为 0 或 1。让我们来为该问题构建一个模型：

# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 层按顺序堆叠以构建分类器：
#
# 第一层是嵌入（Embedding）层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过模型训练学习到的。
# 向量向输出数组增加了一个维度。得到的维度为：(batch, sequence, embedding)。
# 接下来，GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
# 该定长输出向量通过一个有 16 个隐层单元的全连接（Dense）层传输。
# 最后一层与单个输出结点密集连接。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。









































