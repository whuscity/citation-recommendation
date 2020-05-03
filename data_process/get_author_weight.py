"""在获得author的权重时，使用词频作为基准。首先计算全体数据集中作者的词频
在目标文献中作者权重的计算时，使用作者词频然后进行归一化计算，求得目标文献各作者的权重"""
from collections import Counter
import pandas as pd
import json

if __name__ == "__main__":
    filename_path = "../data/aan/aan.csv"
    test_filename_path = "../data/aan/aan_test.csv"
    weight_path = "../data/aan/aan_author_weight.json"
    # 获取全体数据集作者数据
    authors = []
    df = pd.read_csv(filename_path)
    author_ids = list(df['author_id'])
    # 获取目标文献集合作者数据
    test_df = pd.read_csv(test_filename_path)
    test_ids = list(test_df['paper_id'])
    test_ids = list(map(lambda x: str(x), test_ids))
    test_author_ids = list(test_df['author_id'])
    for au_id in author_ids:
        authors += au_id.split(";")
    author_frequency = Counter(authors)
    total_nums = 0
    author_weight = {}
    # 计算目标文献中作者节点权重，归一化处理
    for i, aus in enumerate(test_author_ids):
        author_weight[test_ids[i]] = {}
        per_nums = 0
        if type(aus) == float:
            author_weight[test_ids[i]] = 0
        else:
            for au in aus.split(";"):
                per_nums += author_frequency[au]
            for au in aus.split(";"):
                author_weight[test_ids[i]][au] = author_frequency[au] / per_nums
    json.dump(author_weight, open(weight_path, "w"))
    print("保存完毕")
