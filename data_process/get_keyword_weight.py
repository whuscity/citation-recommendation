"""在获得keyword的权重时，使用tf-idf值作为基准。首先计算全体数据集中keyword的词频
在目标文献中keyword权重的计算时，使用keyword的tf-idf值后进行归一化计算，求得目标文献各keyword的权重"""
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

if __name__ == "__main__":
    filename_path = "../data/aan/aan.csv"
    test_filename_path = "../data/aan/aan_test.csv"
    weight_path = "../data/aan/aan_author_weight.json"
    # 获取全体数据集关键词数据
    df = pd.read_csv(filename_path)
    indexes = []
    paper_ids = list(df['paper_id'])
    paper_ids = list(map(lambda x: str(x), paper_ids))
    # 获取目标文献关键词数据
    test_df = pd.read_csv(test_filename_path)
    test_keywords = list(test_df['keyword_id'])
    test_ids = list(test_df['paper_id'])
    test_ids = list(map(lambda x: str(x), test_ids))
    # 计算tf-idf值流程
    for each in test_ids:
        indexes.append(paper_ids.index(each))
    corpus = list(df['keyword_id'])
    corpus = list(map(lambda x: " ".join(x.split(";")), corpus))
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # toarray在数据集较大时，内存不够会发生溢出
    weight = tfidf.toarray()
    test_weights = []
    for i in indexes:
        test_weights.append(weight[i])
    weight_dict = {}
    for i, w in enumerate(test_weights):
        weight_dict[test_ids[i]] = {}
        for j in range(len(word)):
            if w[j] > 0:
                if word[j] in test_keywords[i].split(";"):
                    weight_dict[test_ids[i]][word[j]] = w[j]
        print("第{}条权重已生成".format(i))
    json.dump(weight_dict, open(weight_path, "w"))
