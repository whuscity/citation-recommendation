"""使用metapath2vec获取节点向量"""
from gensim.models import Word2Vec


def learn_embeddings(walks, model_path, window_size, min_number):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=128, window=window_size, min_count=min_number, sg=1, workers=12,
                     iter=1)
    model.wv.save_word2vec_format(model_path)
    print("保存完毕")


if __name__ == "__main__":
    window_size = 15
    min_number = 0
    walks = []
    walk_path = "../data/aan/aan_p1_15_80.txt"
    emb_path = "../data/aan_metapath2vec_p1_15_80.emb"
    with open(walk_path, "r") as f:
        res = f.readlines()
        for each in res:
            walks.append(each.strip("\n").split(" "))
    learn_embeddings(walks, emb_path, window_size, min_number)
