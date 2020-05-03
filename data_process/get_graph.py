import networkx as nx
def read_txt(filename):
    with open(filename, "r", encoding="utf8") as f:
        res = f.readlines()
        f.close()
    pairs = []
    for edge in res:
        edge_list = edge.strip("\n").split(" ")
        pairs.append((int(edge_list[0]),int(edge_list[1])))
    return pairs
path = nx.DiGraph()
edges = read_txt("new_aan_normal_edges.txt")
path.add_edges_from(edges)
dedges = path.edges()
ans = []
for each in edges:
    if each not in dedges:
        ans.append(each)
nx.write_edgelist(path, './data/aan/new_aan_normal_edges.edgelist')