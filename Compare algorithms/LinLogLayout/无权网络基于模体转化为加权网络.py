# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:47:15 2021

@author: l
"""

import numpy as np
import igraph as ig
import random  
from numpy import random
import networkx as nx
import copy
import pandas as pd
import itertools
from pandas import DataFrame
from numpy import mean,std,median
def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item] 
#计算模体结构3-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def three_one_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        return len(set(u_friends) & set(v_friends))   
#计算模体结构3-2(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/2为模体数量
def three_two_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    u_friends=list(u_friends)
    v_friends=list(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    num = len(u_mor) + len(v_mor)
    return num
#计算模体结构3-2(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/2为模体数量
def three_two_morphology_new(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    u_friends=list(u_friends)
    v_friends=list(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
#    u_mor = list(set(u_friends)-set(v_friends))
#    v_mor = list(set(v_friends)-set(u_friends))
    num = len(v_friends) + len(u_friends)
    return num
#计算模体结构4-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def four_one_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    u_list = []
    v_list = []
    if len(u_mor) <= 1:
        deta1 = 0
    else:
        for i in itertools.combinations(u_mor,2):
            u_list.append(i)
        deta1 = int(len(u_list))
        for p,q in u_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta1 -= 1
            else:
                deta1 += 0
    if len(v_mor) <= 1:
        deta2 = 0
    else:
        for j in itertools.combinations(v_mor,2):
            v_list.append(j)
        deta2 = int(len(v_list))
        for p,q in v_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta2 -= 1
            else:
                deta2 += 0
    return deta1+deta2
#计算模体结构4-2（连边参与模体构造数量）,sum(ij_participate_motif_number_list)/3为模体数量
def four_two_morphology(G,edge_all):
    #求列表的长度
    n=len(edge_all)
    #生成一个由n个0组成的列表
    ij_participate_motif_number_list=[0 for x in range(n)]
    for u,v in edge_all: 
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if u in v_friends:
            v_friends.remove(u)
        if v in u_friends:
            u_friends.remove(v)
        u_mor = list(set(u_friends)-set(v_friends))
        v_mor = list(set(v_friends)-set(u_friends))
        mor_list0 = []
        if (u_mor == []) or (v_mor == []):
            index0=edge_all.index((u,v))
            ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+0
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list0.append((i,j))
            deta = int(len(mor_list0))
            mor_list=copy.deepcopy(mor_list0)
            for p,q in mor_list0:
                if (p,q) in edge_all or (q,p) in edge_all:
                    mor_list.remove((p,q))
                    deta -= 1
            index0=edge_all.index((u,v))
            ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+deta
            #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
            node_list=[]
            node_number_list=[]
            for i in range(len(mor_list)):
                for j in mor_list[i]:
                   node_list.append(j) 
            set_node_list=set(node_list)
            for i in set_node_list:
                node_number_list.append([i,node_list.count(i)])
            #判断每个节点与u、v哪个节点相连接，并更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
            for i in range(len(node_number_list)):
                if (node_number_list[i][0],u) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],u))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (u,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((u,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]      
                elif (node_number_list[i][0],v) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],v))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (v,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((v,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
    return ij_participate_motif_number_list
#计算模体结构4-3(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def four_three_morphology(G,edge_all):
    #求列表的长度
    n=len(edge_all)
    #生成一个由n个0组成的列表
    ij_participate_motif_number_list=[0 for x in range(n)]
    for u,v in edge_all:
        index_uv=edge_all.index((u,v))
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if u in v_friends:
            v_friends.remove(u)
        if v in u_friends:
            u_friends.remove(v)
        u_mor = list(set(u_friends)-set(v_friends))
        v_mor = list(set(v_friends)-set(u_friends))
        u_list0 = []
        v_list0 = []
        if len(u_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            for i in itertools.combinations(u_mor,2):
                u_list0.append(i)
            deta1 = 0
            u_list=[]
            for p,q in u_list0:
                if (p,q) in edge_all:
                    deta1 += 1
                    u_list.append((p,q))
                    index0=edge_all.index((p,q))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                elif (q,p) in edge_all:
                    deta1 += 1
                    u_list.append((p,q))
                    index0=edge_all.index((q,p))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                else:
                    deta1 += 0
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta1
            #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
            node_list=[]
            node_number_list=[]
            for i in range(len(u_list)):
                for j in u_list[i]:
                   node_list.append(j) 
            set_node_list=set(node_list)
            for i in set_node_list:
                node_number_list.append([i,node_list.count(i)])
            #更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
            for i in range(len(node_number_list)):
                if (node_number_list[i][0],u) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],u))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (u,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((u,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]   
        if len(v_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            for i in itertools.combinations(v_mor,2):
                v_list0.append(i)
            deta2 = 0
            v_list=[]
            for p,q in v_list0:
                if (p,q) in edge_all:
                    deta2 += 1
                    v_list.append((p,q))
                    index0=edge_all.index((p,q))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                elif (q,p) in edge_all:
                    deta2 += 1
                    v_list.append((p,q))
                    index0=edge_all.index((q,p))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                else:
                    deta2 += 0
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta2
            #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
            node_list=[]
            node_number_list=[]
            for i in range(len(v_list)):
                for j in v_list[i]:
                   node_list.append(j) 
            set_node_list=set(node_list)
            for i in set_node_list:
                node_number_list.append([i,node_list.count(i)])
            #更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
            for i in range(len(node_number_list)):
                if (node_number_list[i][0],v) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],v))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (v,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((v,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]      
    return ij_participate_motif_number_list
#计算模体结构4-4(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/5为模体数量
def four_four_morphology(G,edge_all):
    #求列表的长度
    n=len(edge_all)
    #生成一个由n个0组成的列表
    ij_participate_motif_number_list=[0 for x in range(n)]
    for u,v in edge_all: 
        index_uv=edge_all.index((u,v))
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
            else:
                cn_edge = []
                for i in itertools.combinations(cn,2):
                    cn_edge.append(i)
                d1 = 0
                #不相互连接的连边集合
                cn_edge0=copy.deepcopy(cn_edge)
                for p,q in cn_edge:
                   if (p,q) in edge_all or (q,p) in edge_all:
                       d1 += 1
                       cn_edge0.remove((p,q))
                   else:
                       d1 += 0
                deta = int(len(cn_edge)) - d1
                ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta
                #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
                node_list=[]
                node_number_list=[]
                for i in range(len(cn_edge0)):
                    for j in cn_edge0[i]:
                       node_list.append(j) 
                set_node_list=set(node_list)
                for i in set_node_list:
                    node_number_list.append([i,node_list.count(i)])
                #判断每个节点与u、v哪个节点相连接，并更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
                for i in range(len(node_number_list)):
                    if (node_number_list[i][0],u) in edge_all:
                        index1=edge_all.index((node_number_list[i][0],u))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                    else :
                        index1=edge_all.index((u,node_number_list[i][0]))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]      
                    if (node_number_list[i][0],v) in edge_all:
                        index1=edge_all.index((node_number_list[i][0],v))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                    else :
                        index1=edge_all.index((v,node_number_list[i][0]))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
    return ij_participate_motif_number_list
#计算模体结构4-5(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def four_five_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    mor_list = []
    if (u_mor == []) or (v_mor == []):
        return 0
    else:
        for i in u_mor:
            for j in v_mor:
                mor_list.append((i,j))
        deta = 0
        for p,q in mor_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta += 1
        return deta
#计算模体结构4-6(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/6为模体数量
def four_six_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        if len(cn) <= 1:
            return 0
        else:
            cn_edge = []
            for i in itertools.combinations(cn,2):
                cn_edge.append(i)
            deta = 0
            for p,q in cn_edge:
               if (p,q) in edge_all or (q,p) in edge_all:
                    deta += 1
               else:
                   deta += 0
            return deta
def moti_num(all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    
    #计算三阶模体 和M4(a,e,f)和S1和SS5
    ij_participate_motif_number_list=[]
    for i,j in all_edge_G:
        ij_participate_motif_number=four_six_morphology(i,j,GG)
        ij_participate_motif_number_list.append(ij_participate_motif_number)
    
    #计算4-2,4-3,4-4 S2-S14 SS1-SS4模体 函数中遍历 
    # ij_participate_motif_number_list=four_five_morphology(GG,all_edge_G)       
    return ij_participate_motif_number_list
# 网络信息

    
G = nx.read_edgelist("football.txt")
G = G.to_undirected()
n=G.number_of_nodes()
#获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist("football.txt")
Gi=Gi.subgraph(map(int,G.nodes()))          
Gi=Gi.as_undirected()
##################################################
edge_all = Gi.get_edgelist()
ij_participate_motif_number_list = moti_num(edge_all)

all_edge_G=copy.deepcopy(edge_all)
for i in range(len(ij_participate_motif_number_list)):
    ij_participate_motif_number_list[i]=ij_participate_motif_number_list[i]+1
    
for i in range(0,len(all_edge_G)):
    all_edge_G[i]=list(all_edge_G[i])
    all_edge_G[i].append(ij_participate_motif_number_list[i])  


new_G=nx.Graph()
new_G.add_weighted_edges_from(all_edge_G) 
nx.write_weighted_edgelist(new_G,'football_m8.el')
