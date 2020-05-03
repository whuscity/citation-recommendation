import networkx as nx

""" get_G_from_edges的作用是获取边的权重"""


def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    # edge_dict获取每个边的权值，比如a,b节点连接两次，那么edge['a_b']=2
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G


"""获得训练集中所有的边,edge_data_by_type依据边的类型来保存边，键值是边的类型，键是'Base'就是所有类型边的集合"""


def load_training_data(f_name):
    print('We are loading training data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    print('Finish loading training data')
    return edge_data_by_type


"""测试集的数据格式为 边的类型 节点1 节点2 是否正确，分为true_edge_data_by_type和false_edge_data_by_type"""


def load_testing_data(f_name):
    print('We are loading testing data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


"""异构时用于保存节点类型"""


def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type