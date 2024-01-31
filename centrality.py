#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:47:34 2020
这个脚本的作用是在全脑的网络内寻找连接最多的中心点，已经和我们的连接组学研究进行适配，可以直接
组合使用
@author: Haosu
"""
import os
import os.path
import pandas as pd
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import math


outpath= '/Users/apple/Desktop/brain_connectome/036'
import os
import os.path

interactive = False

#folder = sys.argv[1] + "/"

folder = os.getcwd()
filepath = folder

filename = filepath.split("/")[-2]

title = filename[0:3]  

connMatrix = pd.read_csv(folder+'/connMatrix.txt', sep='\t')
coords = pd.read_csv(folder+'/tms_coords.txt', sep='\t')
# filter out the row region=='Background'
coords_use = coords.loc[coords.Label!='Background']
label = set(coords_use.Label.unique())
#=====hemi==============
# drop elements contain 'Vermis'
label_filted = [x for x in connMatrix.columns.to_list() if 'Vermis' not in x]
# drop elements contain 'Cerebelum'
label_filted = [x for x in label_filted if 'Cerebelum' not in x]
# drop related rows
connMatrix_filted = connMatrix.loc[connMatrix.Region.isin(label_filted)]
# drop related columns
connMatrix_filted = connMatrix_filted[label_filted]
# For the later negetive matrix without the cerebellum
connHemi = connMatrix_filted
# re-order the columns to get symmetric matrix
cols_filted = connMatrix_filted.Region.to_list()
# add 'Region' at the beginning
#cols_filted.insert(0, 'Region')
# re-order columns
connMatrix_filted = connMatrix_filted[cols_filted]
connMatrix_filted_h = np.array(connMatrix_filted, dtype = float)
##矩阵转为Graph的方法，是Binary的
G = nx.Graph()
for i in range(len(connMatrix_filted_h)):
    for j in range(len(connMatrix_filted_h)):
        if connMatrix_filted_h[i,j] > 2:
            G.add_edge(i, j)
"""
矩阵转为Graph的方法，是加权的。If create_using is networkx.MultiGraph or networkx.
MultiDiGraph, parallel_edges is True, and the entries of A are of type int, then
this function returns a multigraph (of the same type as create_using) with 
parallel edges.
"""

def progressive_widening_search(G, source, value, condition, initial_width=1):
    """Progressive widening beam search to find a node.

    The progressive widening beam search involves a repeated beam
    search, starting with a small beam width then extending to
    progressively larger beam widths if the target node is not
    found. This implementation simply returns the first node found that
    matches the termination condition.

    `G` is a NetworkX graph.

    `source` is a node in the graph. The search for the node of interest
    begins here and extends only to those nodes in the (weakly)
    connected component of this node.

    `value` is a function that returns a real number indicating how good
    a potential neighbor node is when deciding which neighbor nodes to
    enqueue in the breadth-first search. Only the best nodes within the
    current beam width will be enqueued at each step.

    `condition` is the termination condition for the search. This is a
    function that takes a node as input and return a Boolean indicating
    whether the node is the target. If no node matches the termination
    condition, this function raises :exc:`NodeNotFound`.

    `initial_width` is the starting beam width for the beam search (the
    default is one). If no node matching the `condition` is found with
    this beam width, the beam search is restarted from the `source` node
    with a beam width that is twice as large (so the beam width
    increases exponentially). The search terminates after the beam width
    exceeds the number of nodes in the graph.

    """
    # Check for the special case in which the source node satisfies the
    # termination condition.
    if condition(source):
        return source
    # The largest possible value of `i` in this range yields a width at
    # least the number of nodes in the graph, so the final invocation of
    # `bfs_beam_edges` is equivalent to a plain old breadth-first
    # search. Therefore, all nodes will eventually be visited.
    log_m = math.ceil(math.log2(len(G)))
    for i in range(log_m):
        width = initial_width * pow(2, i)
        # Since we are always starting from the same source node, this
        # search may visit the same nodes many times (depending on the
        # implementation of the `value` function).
        for u, v in nx.bfs_beam_edges(G, source, value, width):
            if condition(v):
                return v
    # At this point, since all nodes have been visited, we know that
    # none of the nodes satisfied the termination condition.
    raise nx.NodeNotFound("no node satisfied the termination condition")
#from networkx.convert_matrix import from_numpy_array
#G = from_numpy_array(connMatrix_filted_h, parallel_edges=False, create_using=None)

#特征向量中心度（Eigenvector Centrality）是终止于节点 i 的长度为无穷的游走的数量。
centrality = nx.eigenvector_centrality(G)
avg_centrality = sum(centrality.values()) / len(G)


def has_high_centrality(v):
    return centrality[v] >= avg_centrality

print (max(centrality.values()))

source = 0 #表示的是要去寻找0号点接触的centrality最高的点！！！
value = centrality.get
condition = has_high_centrality

found_node = progressive_widening_search(G, source, value, condition)
#found_node所找到的点是与source连接的点中，centrality最高的位置！！！
c = centrality[found_node]
print(f"found node {found_node} with centrality {c}")

pos = nx.spring_layout(G)
options = {
    "node_color": "blue",
    "node_size": 20,
    "line_color": "grey",
    "linewidths": 0,
    "width": 0.1,
}
nx.draw(G, pos,with_labels=True, **options)
#nx.draw(G, pos, **options)
# Draw node with high centrality among all nodes related to the source node 
# mark it to be large and red
nx.draw_networkx_nodes(G, pos, nodelist=[found_node], node_size=200, node_color="r",dpi=300)
plt.show()

"""
对图性质的描述分析
"""
degree_sequence = list(G.degree())
#计算边的数量，但也计算度序列的度量：
nb_arr = len(G.edges())
avg_degree = np.mean(np.array(degree_sequence)[:,1])
med_degree = np.median(np.array(degree_sequence)[:,1])
max_degree = max(np.array(degree_sequence)[:,1])
min_degree = np.min(np.array(degree_sequence)[:,1])
#最后，打印所有信息：
print("Number of edges : " + str(nb_arr))
print("Maximum degree : " + str(max_degree))
print("Minimum degree : " + str(min_degree))
print("Average degree : " + str(avg_degree))
print("Median degree : " + str(med_degree))


"""
为寻找centrality最高位点的位置
"""
Max = max(centrality.values())

result_max = max(centrality,key=lambda x:centrality[x])
print(f'max:{result_max}')

source = result_max 

value = centrality.get
condition = has_high_centrality

found_node = progressive_widening_search(G, source, value, condition)
#found_node所找到的点是与source连接的点中，centrality最高的位置！！！
c = centrality[found_node]
print(f"found node {found_node} with centrality {c}")

pos = nx.spring_layout(G)
options = {
    "node_color": "blue",
    "node_size": 20,
    "line_color": "grey",
    "linewidths": 0,
    "width": 0.1,
}
nx.draw(G, pos,with_labels=True, **options)
#nx.draw(G, pos, **options)

# Draw node with high centrality as large and red
nx.draw_networkx_nodes(G, pos, nodelist=[found_node], node_size=200, node_color="r",dpi=300)
plt.show()

"""
不同的中心度向量代表的含义是不同的
度中心度（Degree Centrality）统计的是终止于节点 i 的长度为 1 的游走的数量。
    这能够衡量传入和传出关系。这能通过 C(Xi)=di 给出。度中心度可用于识别社交网络中最有影响力的人。
特征向量中心度（Eigenvector Centrality）是终止于节点 i 的长度为无穷的游走的数量。
    这能让有很好连接相邻节点的节点有更高的重要度
接近度中心度（Closeness Centrality）检测的是可以在图中有效传播信息的节点。
    这可用于识别假新闻账户或恐怖分子，以便隔离能向图中其它部分传播信息的个体。
居间性中心度（Betweenness Centrality）检测的是节点在图中的信息流上所具有的影响量。
    这通常可用于发现用作从图的一部分到另一部分的桥的节点，比如用在电信网络的数据包传递处理器或假新闻传播分析中。
"""
"""
page rank函数的应用，在于发现如果出现某点消失后，对于整个信息传递链的影响程度
"""
page_rank=nx.pagerank(G,alpha=0.9)
page_rank = list(page_rank.values())

plt.figure(figsize=(18, 12))# Degree Centrality
f, axarr = plt.subplots(1, 1, num=1)
nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = page_rank, node_size=300, pos=pos, with_labels=True)
axarr.set_title('Pagerank Centrality', size=16)# Betweenness Centrality

c_degree = nx.degree_centrality(G)
c_degree = list(c_degree.values())

c_eigenvector = nx.eigenvector_centrality(G)
c_eigenvector = list(c_eigenvector.values())

c_closeness = nx.closeness_centrality(G)
c_closeness = list(c_closeness.values())

c_betweenness = nx.betweenness_centrality(G)
c_betweenness = list(c_betweenness.values())

# Plot the centrality of the nodes
plt.figure(figsize=(18, 12))# Degree Centrality
f, axarr = plt.subplots(2, 2, num=1)
plt.sca(axarr[0,0])
nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_degree, node_size=300, pos=pos, with_labels=True)
axarr[0,0].set_title('Degree Centrality', size=16)# Degree Centrality
plt.sca(axarr[0,1])
nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_eigenvector, node_size=300, pos=pos, with_labels=True)
axarr[0,1].set_title('Eigenvalue Centrality', size=16)# Eigenvalue Centrality
plt.sca(axarr[1,0])
nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_closeness, node_size=300, pos=pos, with_labels=True)
axarr[1,0].set_title('Closeness Centrality', size=16)# Closeness Centrality
plt.sca(axarr[1,1])
nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_betweenness, node_size=300, pos=pos, with_labels=True)
axarr[1,1].set_title('Betweenness Centrality', size=16)# Betweenness Centrality

#plt.figure(figsize=(18, 12))# Degree Centrality
#f, axarr = plt.subplots(2, 2, num=1)
#plt.sca(axarr[0,0])
#nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_degree, node_size=300, pos=pos)
#axarr[0,0]# Degree Centrality
#plt.sca(axarr[0,1])
#nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_eigenvector, node_size=300, pos=pos)
#axarr[0,1]# Eigenvalue Centrality
#plt.sca(axarr[1,0])
#nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_closeness, node_size=300, pos=pos)
#axarr[1,0]# Closeness Centrality
#plt.sca(axarr[1,1])
#nx.draw(G, cmap = plt.get_cmap('inferno'), node_color = c_betweenness, node_size=300, pos=pos)
#axarr[1,1]# Betweenness Centrality

