"""引文推荐效果评估函数"""
import json
import pandas as pd
import math
import numpy as np


# 计算节点向量的余弦相似度
def get_score(local_model, node1, node_features):
    try:
        vector1 = np.array(local_model[node1])
        vector2 = np.array(node_features)
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        vector1 = np.random.uniform(low=-1.0, high=1.0, size=128)
    vector2 = np.array(node_features)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# 用于LINE模型训练出的json格式将字符串转化为float格式
def str_to_float(str_model):
    float_model = {}
    for key, value in str_model.items():
        float_model[key] = np.array(value)
    return float_model


# 根据author_weight和keyword_weight获得目标文献节点向量
def get_paper_embeddings(model, df, node, mode, author_weight_path, keyword_weight_path):
    keyword_weights = 0
    data_author_weight = json.load(open(author_weight_path, "r"))
    data_keyword_weight = json.load(open(keyword_weight_path, "r"))
    kw_features = np.zeros(128)
    au_features = np.zeros(128)
    keywords = list(df[df['paper_id'] == int(node)]['keyword_id'])[0]
    authors = list(df[df['paper_id'] == int(node)]['author_id'])[0]
    if type(keywords) == float or len(keywords) == 0:
        kw_features = np.zeros(128)
    else:
        keywords_item = keywords.split(";")
        for item in keywords_item:
            keyword_weights += data_keyword_weight[node][item]
        for kw in keywords_item:
            try:
                kw_features += (data_keyword_weight[node][kw] / keyword_weights) * model[kw]
            except Exception as e:
                pass
    if type(authors) == float or len(authors) == 0:
        au_features = np.zeros(128)
    else:
        authors_item = authors.split(";")
        for au in authors_item:
            try:
                au_features += data_author_weight[node][au] * model[au]
            except Exception as e:
                pass
    if mode == "add":
        node_features = (kw_features + au_features) / 2
    else:
        node_features = kw_features * au_features

    if (np.array(node_features) == np.zeros(128)).all():
        node_features = np.random.uniform(low=-1.0, high=1.0, size=128)

    return node_features


# 获取目标文献节点向量
def get_testnodes(csv_path, edges_path):
    df = pd.read_csv(csv_path)
    paper_name = df['id']
    paper_id = df['paper_id']
    node_dict = {}
    p_p_edges = []
    test_df = df[df['year'] > 2013]
    test_paper_name = list(test_df['id'])
    for x, y in zip(paper_name, paper_id):
        node_dict[x] = y

    with open(edges_path, "r") as f:
        edges = f.readlines()
    for each in edges:
        p_p_edges.append(each.strip("\n"))
    return test_paper_name, p_p_edges, node_dict


# NDCG函数
def get_NDCG_n(reli, n):
    numbers = [i for i in range(1, n + 1)]
    log_numbers = []
    for number in numbers:
        log_numbers.append(math.log(number + 1, 2))
    DCG_list = []
    for x, y in zip(reli, log_numbers):
        DCG_list.append(x / y)
    IDCG_list = []
    for x, y in zip(sorted(reli, reverse=True), log_numbers):
        IDCG_list.append(x / y)
    return sum(DCG_list) / (sum(IDCG_list) + 0.00001)


# 根据节点向量获取推荐的文章节点
def get_most_similar_node(test_nodes, model, df, mode, vocab, papers, author_weight_path, keyword_weight_path):
    results = dict()
    for j, node in enumerate(test_nodes):
        result = {}
        # for i in range(14318):
        #     score = get_score(model, node, str(i))
        #     result[str(i)] = score
        node_features = get_paper_embeddings(model, df, node, mode, author_weight_path, keyword_weight_path)
        for candidate_paper in papers:
            score = get_score(model, candidate_paper, node_features)
            result[candidate_paper] = score
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        results[node] = sorted_result[:100]
        print("第{0}个节点已找到相似度最高的前100个点".format(j))
    return results


# 获取不同引文数量的目标文献
def choose_K_true_cites(k, true_json):
    K_dict = {}
    for key, v in true_json.items():
        if len(v) >= k:
            K_dict[key] = v
    return K_dict


# 引文推荐效果评估
def get_metric(true_cites, predict_cites):
    predictions = {}
    precision_25_list, precision_50_list, precision_75_list, precision_100_list = [], [], [], []
    recall_25_list, recall_50_list, recall_75_list, recall_100_list = [], [], [], []
    ndcg_25_list, ndcg_50_list, ndcg_75_list, ndcg_100_list = [], [], [], []
    for key, value in predict_cites.items():
        values = []
        for each_node in value:
            values.append(each_node[0])
        predictions[key] = values
    for key, value in true_cites.items():
        pred_25, pred_50, pred_75, pred_100 = predictions[key][:25], predictions[key][:50], predictions[key][:75], \
                                              predictions[key][:100]
        count_25, count_50, count_75, count_100 = 0, 0, 0, 0
        ndcg_25, ndcg_50, ndcg_75, ndcg_100 = [0] * 25, [0] * 50, [0] * 75, [0] * 100
        for node in true_cites[key]:
            if node in pred_25:
                count_25 += 1
                ndcg_25[pred_25.index(node)] = 1
            if node in pred_50:
                count_50 += 1
                ndcg_50[pred_50.index(node)] = 1
            if node in pred_75:
                count_75 += 1
                ndcg_75[pred_75.index(node)] = 1
            if node in pred_100:
                count_100 += 1
                ndcg_100[pred_100.index(node)] = 1
        precision_25 = count_25 / 25
        precision_50 = count_50 / 50
        precision_75 = count_75 / 75
        precision_100 = count_100 / 100
        recall_25 = count_25 / len(true_cites[key])
        recall_50 = count_50 / len(true_cites[key])
        recall_75 = count_75 / len(true_cites[key])
        recall_100 = count_100 / len(true_cites[key])
        precision_25_list.append(precision_25)
        precision_50_list.append(precision_50)
        precision_75_list.append(precision_75)
        precision_100_list.append(precision_100)
        recall_25_list.append(recall_25)
        recall_50_list.append(recall_50)
        recall_75_list.append(recall_75)
        recall_100_list.append(recall_100)
        ndcg_25_list.append(get_NDCG_n(ndcg_25, 25))
        ndcg_50_list.append(get_NDCG_n(ndcg_50, 50))
        ndcg_75_list.append(get_NDCG_n(ndcg_75, 75))
        ndcg_100_list.append(get_NDCG_n(ndcg_100, 100))

    average_precision_25 = sum(precision_25_list) / len(precision_25_list)
    average_precision_50 = sum(precision_50_list) / len(precision_50_list)
    average_precision_75 = sum(precision_75_list) / len(precision_75_list)
    average_precision_100 = sum(precision_100_list) / len(precision_100_list)
    average_recall_25 = sum(recall_25_list) / len(recall_25_list)
    average_recall_50 = sum(recall_50_list) / len(recall_50_list)
    average_recall_75 = sum(recall_75_list) / len(recall_75_list)
    average_recall_100 = sum(recall_100_list) / len(recall_100_list)
    average_ndcg_25 = sum(ndcg_25_list) / len(ndcg_25_list)
    average_ndcg_50 = sum(ndcg_50_list) / len(ndcg_50_list)
    average_ndcg_75 = sum(ndcg_75_list) / len(ndcg_75_list)
    average_ndcg_100 = sum(ndcg_100_list) / len(ndcg_100_list)
    precision_top = [average_precision_25, average_ndcg_50, average_ndcg_75, average_precision_100]
    recall_top = [average_recall_25, average_recall_50, average_recall_75, average_recall_100]
    ndcg_top = [average_ndcg_25, average_ndcg_50, average_ndcg_75, average_ndcg_100]

    print("--------------------------Results--------------------------\n")
    print("Top25 metrics: precision is {0}; recall is {1}; ndcg is {2} ".format(average_precision_25, average_recall_25,
                                                                                average_ndcg_25))
    print("Top50 metrics: precision is {0}; recall is {1}; ndcg is {2} ".format(average_precision_50, average_recall_50,
                                                                                average_ndcg_50))
    print("Top75 metrics: precision is {0}; recall is {1}; ndcg is {2} ".format(average_precision_75, average_recall_75,
                                                                                average_ndcg_75))
    print("Top100 metrics: precision is {0}; recall is {1}; ndcg is {2} ".format(average_precision_100,
                                                                                 average_recall_100,
                                                                                 average_ndcg_100))
    return precision_top, recall_top, ndcg_top


if __name__ == "__main__":
    author_weight_path = "../data/aan/aan_author_weight.json"
    keyword_weight_path = "../data/aan/aan_author_weight.json"
    true_cites_path = r"../data/aan/aan_true_cites.json"
    emb_path = '../embs/aan_line.json'
    train_path = "../data/aan/aan_train.csv"
    test_path = "../data/aan_test.csv"
    true_cites = json.load(open(true_cites_path, "r"))
    true_cites = choose_K_true_cites(1, true_cites)
    useful_nodes = list(true_cites.keys())
    print(len(useful_nodes))
    model = json.load(open(emb_path, 'r'))
    model = str_to_float(model)

    train_df = pd.read_csv(train_path)
    papers = list(train_df['paper_id'])
    papers = list(map(lambda x: str(x), papers))

    test_df = pd.read_csv(test_path)

    predict_cites = get_most_similar_node(useful_nodes, model, test_df, "add", papers, author_weight_path,
                                          keyword_weight_path)

    get_metric(true_cites, predict_cites)
