# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:41:14 2020

@author: l
"""
import networkx as nx
#import igraph as ig
#将节点进行重新标号，并将改变完的节点序号写入文件
G= nx.read_edgelist(r'network92.txt')
G=G.to_undirected()
l1 = sorted(list(map(int,G.nodes())))#原始节点 排序
l2 = range(0,G.number_of_nodes())#新节点 排序


#新老节点一一对应
nodes = dict(map(lambda x,y:[x,y],l1,l2)) 
print(nodes)
edge_list=[]
for u,v in G.edges():
   edge_list.append((nodes[int(u)],nodes[int(v)]))
print(edge_list)
new_G=nx.Graph()
new_G.add_edges_from(edge_list)
nx.write_edgelist(new_G,'network92_0.txt',data=False)
