1、本项目主要包括metapath2vec/node2vec/LINE/deepWalk模型，基于这些模型可获得网络中各节点的低维度向量表示
2、data文件夹包括aan/med/dblp三种数据集，每种数据集中包括不带边的类型的节点对文件、带边的类型的节点对文件、网络中节点类型文件
3、dataprocess文件夹主要包括基于元路径获取节点路径的walk.py文件和在node2vec模型中由节点对生成graph格式的get_graph.py，其中get_author_weight.py和get_keyword_weight.py是获得作者和关键词节点权重
4、get_aan.ipynb/get_medicine.ipynb/get_dblp.ipynb是将包含文章ID、关键词、作者、引文信息处理成为输入到模型中的数据。
5、example文件夹中主要包括上述几种模型的运行文件，运行这几个文件可以分别得到不同模型下的节点向量
6、ge文件夹中主要包括各模型的原始代码。
7、输入数据文件格式：对于不带边类型的文件，其每行的格式为“节点1 节点2”，比如“13421 56734”；对于带边的类型的文件，其每行的格式为“边类型 节点1 节点2”，比如“3 13421 56734”
8、对于输出的文件除了LINE输出的是json格式文件以外，metapath2vec/node2vec/deepwalk输出的均为emb文件
