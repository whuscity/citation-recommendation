"""使用LINE模型获得节点向量"""
import sys

sys.path.append(r'D:\pythonplaces\citation-recommendation')
from ge import LINE
import networkx as nx

if __name__ == "__main__":
    G = nx.read_edgelist('../data/aan/new_aan_normal_edges.txt',
                         create_using=nx.Graph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=128, order='first')
    model.train(batch_size=1024, epochs=15, verbose=2)
    embeddings = model.get_embeddings(emb_filepath="../embs/aan_line.json")
