"""使用deepwalk模型获得节点向量"""
import sys

sys.path.append(r'D:\pythonplaces\citation-recommendation')

from ge import DeepWalk

import networkx as nx

if __name__ == "__main__":
    G = nx.read_edgelist('../data/aan/aan_normal_train.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=5, num_walks=10, workers=12)
    model.train(embed_size=128, window_size=5, iter=3, emb_filepath="../embs/aan_deepwalk_test.emb")
    embeddings = model.get_embeddings()
