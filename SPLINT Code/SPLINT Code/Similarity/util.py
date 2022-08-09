import jieba
import re
import numpy as np
import math


def get_stopwords(language):
    return [line.strip() for line in
            open('Similarity/stopwords/{}_stopwords.txt'.format(language), encoding='UTF-8').readlines()]


def preprocess(documents):
    # 使用jieba库进行分词
    text_filter = []
    stopwords_zh = get_stopwords('chinese')
    stopwords_en = get_stopwords('english')
    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-',
                    '，', '。', '：', '；', '？', '（', '）', '【', '】', '！', '￥', '<', '>', '《', '》', ' ', '`', '=',
                    '\n', '/', '\t', '\r', "'", '+']
    for document in documents:
        texts = []
        words = jieba.cut(document)
        for word in words:  # 去除停用词，标点符号，空格换行制表符等
            word = re.sub(r'^\d+$', ' ', word)
            word = word.lower()
            if word not in stopwords_en and word not in stopwords_zh and word not in punctuations:
                texts.append(word)
        text_filter.append(texts)

    # st = LancasterStemmer()
    #     # texts_stemmed = [[st.stem(word) for word in document] for document in text_filter]

    return text_filter


def mean(arr):
    arr = np.asarray(arr)
    cnt = 0
    sum = 0
    for x in arr:
        if x == np.inf or math.isnan(x):
            cnt += 1
        else:
            sum += x

    return sum / (len(arr) - cnt)


def var(arr):
    arr = np.asarray(arr)
    cnt = 0
    sum = 0
    avg = mean(arr)
    for x in arr:
        if x == np.inf or math.isnan(x):
            cnt += 1
        else:
            sum += (x - avg) * (x - avg)

    return sum / (len(arr) - cnt)
