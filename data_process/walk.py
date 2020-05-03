import random
from utils import *


"""基于随机游走模型+元路径获取节点路径"""
class RWGraph():
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        # Simulate a random walk starting from start node.
        # schema为元路径的格式
        # 获取以某个节点开始的元路径
        G = self.G
        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
        #     assert schema_items[0] == schema_items[-1]


            # 确保schema的开始节点和末尾节点是一致的

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():
                if schema_items[0] != schema_items[-1]:

                    if self.node_type[node] == schema_items[len(walk) % len(schema_items)]:
                        candidates.append(node)
                else:
                    if self.node_type[node] == schema_items[len(walk) % (len(schema_items)-1)]:
                        candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length, schema_list=None):
        # 获取整个所有节点的路径
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema_list == None:
                    walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_list))
                # 当需要使用多个元路径时则需要在此处加循环
                else:
                    for i,schema in enumerate(schema_list):
                        if schema.split('-')[0] == self.node_type[node]:
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema))


        return walks
def generate_walks(network_data,num_walks,walk_length,schema_list,node_type_path):
    base_network = network_data['Base']

    if schema_list is not None:
        node_type = load_node_type(node_type_path)
    else:
        node_type = None

    base_walker = RWGraph(get_G_from_edges(base_network),node_type=node_type )
    base_walks = base_walker.simulate_walks(num_walks, walk_length,schema_list=schema_list )


    print('finish generating the walks')

    # return base_walks, all_walks
    return base_walks

if __name__ == "__main__":
    data_path='../data/aan/aan_normal_edges.txt'
    walk_path1="../data/aan/aan_p1_15_80.txt"
    walk_path2="../data/aan/aan_p2_15_80.txt"
    training_data_by_type = load_training_data(data_path)
    # num_walks walk_length是两个非常重要的参会素，控制经过某一节点的路径个数和路径长度
    num_walks = 15
    walk_length = 80
    node_type_path="../data/aan/aan_node_type.txt"
    # 元路径可以自定义
    #schema_list = ['P-K-P-P-K-P','P-A-P','P-P-P','P-A-P-K-P-A-P','P-A-P-A-P']
    schema_list = ['P-P','A-P','P-K']
    # schema_list = ['P-P','P-K']
    # schema_list1 = ['P-P-P','P-K-P','K-P-K', 'K-P-P']
    # schema_list1 = ['P-A-P','A-P-P','A-P-A','P-P-P']
    schema_list1 = ['P-P-P', 'P-A-P', 'P-K-P', 'A-P-A', 'K-P-K', 'K-P-A', 'A-P-K', 'K-P-P', 'A-P-P']
    all_walks = generate_walks(training_data_by_type, num_walks, walk_length,schema_list,node_type_path)
    with open(walk_path1,
              "w", encoding="utf8") as f:
        for walk in all_walks:
            f.write(" ".join(walk) + "\n")
        f.close()
    print("保存完毕")
    new_all_walks = generate_walks(training_data_by_type, num_walks, walk_length, schema_list1)
    with open(
            walk_path2,
            "w", encoding="utf8") as f:
        for walk in new_all_walks:
            f.write(" ".join(walk) + "\n")
        f.close()
    print("保存完毕")