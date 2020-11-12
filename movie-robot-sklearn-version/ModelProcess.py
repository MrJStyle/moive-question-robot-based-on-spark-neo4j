import re
import os
import pickle
import time

import jieba
import jieba.posseg
import numpy as np
from numpy import array as Array

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score


data_dir = './question/'
vocabularies = './vocabulary.txt'
genre_dir = "./自定义词典/genreDict.txt"
movie_dir = "./自定义词典/movieDict.txt"

with open(genre_dir, encoding='utf-8-sig') as f:
    genre_dir = f.read().splitlines()
    for i, line in enumerate(genre_dir):
        genre_dir[i] = line

with open(movie_dir, encoding='utf-8-sig') as f:
    movie_dir = f.read().splitlines()
    for i, line in enumerate(movie_dir):
        movie_dir[i] = line


# 对数据进行分词处理，对于电影、类型采用自定义词典
def pre_possegment(query):
    sentence_seged = jieba.posseg.cut(query)
    outstr = ''
    for x in sentence_seged:
        if x.flag =='nr':
            outstr += "{},".format('nnt')
        elif x.word in movie_dir:
            outstr += "{},".format('nm')
        elif x.word in genre_dir:
            outstr += "{},".format('ng')
        else:
            outstr += "{},".format(x.word)
    outstr = outstr.split(',')
    return outstr


class NaiveBayesModel:
    features: Array
    labels: Array

    model: MultinomialNB

    def __init__(self):
        self.vocabularies = []
        self.model_path = "./nb_model"

    def load(self, data_directory, vocabularies):
        # 读取vocabularies文件，构建高频的词库
        with open(vocabularies, encoding='utf-8') as f:
            vocabulary = f.read().splitlines()
            for i, line in enumerate(vocabulary):
                vocabulary[i] = line.split(':')[-1]
        # print(vocabulary[0])
        texts = []
        self.x = []
        self.vocabularies = vocabulary
        # 遍历读取文件夹中的数据文件，共14个文件
        for parent, dirnames, filenames in os.walk(data_directory):
            self.x = []
            for filename in filenames:
                if filename[0] != '.':
                    # print(os.path.join(data_dir, filename))
                    with open(os.path.join(data_directory, filename), encoding='UTF-8') as f:
                        file = f.read().splitlines()
                    texts.append(file)
                    self.x.append(filename)
        documents = {}
        # print(len(self.x))
        # 对每一个数据文件中的词进行分词操作，并将其文件名作为字典索引的key
        for i, text in enumerate(texts):
            lines = []
            for line in text:
                word = jieba.cut(line)
                lines.append(' '.join(word))
            documents[self.x[i]] = lines

        document_new = {}
        # 对分词后的数据文件的每一行进行和高频词库的匹配，将数据文件的每一行转为一个行向量
        features, labels = [], []

        for i in range(len(documents)):
            document = documents[self.x[i]]

            x = np.zeros((len(document), len(vocabulary)))
            y = np.repeat([i], len(document))

            for row, line in enumerate(document):
                line = line.split(" ")
                for word in line:
                    if word in vocabulary:
                        index = vocabulary.index(word)
                        x[row][index] = 1
            features.append(x)
            labels.append(y)

        self.features = np.concatenate(features, axis=0)
        self.labels = np.concatenate(labels, axis=0)

    def fit(self):
        naive_bayes: MultinomialNB = MultinomialNB(alpha=0.1)
        k_fold = KFold(n_splits=5, shuffle=True)

        accuracy = []
        recall = []
        f1 = []

        for train_index, test_index in k_fold.split(self.features, self.labels):
            naive_bayes.fit(self.features[train_index], self.labels[train_index])

            y_pred = naive_bayes.predict(self.features[test_index])

            accuracy.append(accuracy_score(self.labels[test_index], y_pred))
            recall.append(recall_score(self.labels[test_index], y_pred, average="micro"))
            f1.append(f1_score(self.labels[test_index], y_pred, average="micro"))

        average_accuracy = np.mean(accuracy)
        average_recall = np.mean(recall)
        average_f1 = np.mean(f1)

        print(f"accuracy: {average_accuracy}\nrecall: {average_recall}\nf1-score: {average_f1}")
        self.model = naive_bayes

    def test(self, sentence, vocabularies):
        # sentence = ' '.join(jieba.cut(sentence))
        sentence = sentence.split(" ")
        print('句子抽象化后的结果: {}'.format(sentence))
        vector = [0 for x in range(len(vocabularies))]
        for word in sentence:
            if word in vocabularies:
                index = vocabularies.index(word)
                vector[index] = 1

        vector = np.array(vector)[np.newaxis, :]
        pred = self.model.predict(vector)


        print('The model index is: {}'.format(result.prediction))
        return int(result.prediction)


jieba.load_userdict(movie_dir)
pkl_file = open("vocabulary.pkl", "rb")
vocabulary = pickle.load(pkl_file)
sentence = '不能说的秘密中参演的演员都有哪些'
sentence = pre_possegment(sentence)
sentence = ' '.join(sentence)

start = time.time()
model = NaiveBayesModel()
model.load(data_directory=data_dir, vocabularies=vocabularies)
model.fit()
model.test(sentence, vocabulary)
# end = time.time()
# time = end - start
# print("测试时间为: {:.4f}".format(time))
