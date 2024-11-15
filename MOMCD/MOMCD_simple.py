# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:00:17 2021

@author: l
"""
#	算法4：邻居社区总权重
#	Motif-SOSCDNW (优化Qw, WBias + clean_up_NW + local_search) 

#exetime = 10
#best_Qw = 0.4838408776166325 with Q = 0.41510519395134776 with nmi = 0.707135418720364


import numpy as np
import igraph as ig
import random  
from numpy import random
import networkx as nx
import copy
import pandas as pd
import itertools
import random  as rd
from pandas import DataFrame
from numpy import mean,std,median
#import matplotlib.pyplot as plt
from math import exp,log,sqrt 

#其中有后缀motifadd的加权是用于社区修正中，考虑了模体邻居节点
#计算模体结构3-a(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def three_one_morphology(G,edge_all):
    ij_participate_motif_number_list=[]    
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        u_friends=list(u_friends)
        v_friends=list(v_friends)
        if (u_friends == []) or (v_friends == []):
            ij_participate_motif_number_list.append(0)
        else:
            ij_participate_motif_number_list.append(len(set(u_friends) & set(v_friends)))
    return  ij_participate_motif_number_list
def three_one_morphology_motifadd(G,edge_all):
    n=len(G.nodes())
    motif_matrix=np.zeros((n,n))
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        u_friends=list(u_friends)
        v_friends=list(v_friends)
        if (u_friends == []) or (v_friends == []):
            pass
        else:
            motif_matrix[u][v]+=len(set(u_friends) & set(v_friends))
            motif_matrix[v][u]+=len(set(u_friends) & set(v_friends))
    return  motif_matrix 
#计算模体结构3-2(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/2为模体数量
def three_two_morphology(G,edge_all):
    ij_participate_motif_number_list=[]    
    for u,v in edge_all:      
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
        ij_participate_motif_number_list.append(len(u_mor) + len(v_mor))
    return ij_participate_motif_number_list
def three_two_morphology_motifadd(G,edge_all):
    n=len(G.nodes())
    motif_matrix=np.zeros((n,n))    
    for u,v in edge_all:      
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
        for j in u_mor:
            motif_matrix[j][v]+=1
            motif_matrix[v][j]+=1
        for j in v_mor:
            motif_matrix[j][u]+=1
            motif_matrix[u][j]+=1        
        motif_matrix[u][v]+=len(u_mor) + len(v_mor)
        motif_matrix[v][u]+=len(u_mor) + len(v_mor)
    return motif_matrix
#计算模体结构4-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def four_one_morphology(G,edge_all):    
    ij_participate_motif_number_list=[]    
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
        u_list = []
        v_list = []
        if len(u_mor) <= 1:
            deta1 = 0
        else:
            for i in itertools.combinations(u_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    u_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta1 = int(len(u_list))               
        if len(v_mor) <= 1:
            deta2 = 0
        else:
            for i in itertools.combinations(v_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    v_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta2 = int(len(v_list))
        ij_participate_motif_number_list.append(deta2+deta1)
    return ij_participate_motif_number_list
def four_one_morphology_motifadd(G,edge_all):
    #求列表的长度
    #生成全0矩阵
    n=len(G.nodes())
    motif_matrix=np.zeros((n,n)) 
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
        u_list = []
        v_list = []
        if len(u_mor) <= 1:
            deta1 = 0
        else:
            for i in itertools.combinations(u_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    u_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta1 = int(len(u_list)) 
              
        if len(v_mor) <= 1:
            deta2 = 0
        else:
            for i in itertools.combinations(v_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    v_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta2 = int(len(v_list))
        motif_matrix[u][v]=deta2+deta1
        motif_matrix[v][u]=deta2+deta1
        for i in u_list:
            motif_matrix[i[0]][i[1]]+=1/3
            motif_matrix[i[1]][i[0]]+=1/3
            motif_matrix[v][i[0]]+=1/3
            motif_matrix[i[0]][v]+=1/3
            motif_matrix[v][i[1]]+=1/3
            motif_matrix[i[1]][v]+=1/3
        for i in v_list:
            motif_matrix[i[0]][i[1]]+=1/3
            motif_matrix[i[1]][i[0]]+=1/3
            motif_matrix[u][i[0]]+=1/3
            motif_matrix[i[0]][u]+=1/3
            motif_matrix[u][i[1]]+=1/3
            motif_matrix[i[1]][u]+=1/3        
    return motif_matrix
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
#计算模体结构4-2（连边参与模体构造数量）,sum(ij_participate_motif_number_list)/3为模体数量
def four_two_morphology_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
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
            motif_matrix[u][v]+=0
            motif_matrix[v][u]+=0
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
            motif_matrix[u][v]+=deta
            motif_matrix[v][u]+=deta
            for mor_list_i in mor_list:
                motif_matrix[mor_list_i[0]][mor_list_i[1]]+=1
                motif_matrix[mor_list_i[1]][mor_list_i[0]]+=1
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
                node_i=node_number_list[i][0]
                if (min(node_i,u),max(node_i,u)) in edge_all:
                    motif_matrix[node_i][u]+=node_number_list[i][1]
                    motif_matrix[u][node_i]+=node_number_list[i][1] 
                    motif_matrix[node_i][v]+=node_number_list[i][1]
                    motif_matrix[v][node_i]+=node_number_list[i][1]                     
                elif (min(node_i,v),max(node_i,v)) in edge_all:
                    motif_matrix[node_i][v]+=node_number_list[i][1]
                    motif_matrix[v][node_i]+=node_number_list[i][1]
                    motif_matrix[node_i][u]+=node_number_list[i][1]
                    motif_matrix[u][node_i]+=node_number_list[i][1]
    return motif_matrix
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
        #如果节点u的邻居节点除v外只有一个，那么无法构成一个三角形，因此无模体
        if len(u_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(u_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    u_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in u_list0:
                ij_participate_motif_number_list[edge_all.index((min(u,p),max(u,p)))]=ij_participate_motif_number_list[edge_all.index((min(u,p),max(u,p)))]+1
                ij_participate_motif_number_list[edge_all.index((min(u,q),max(u,q)))]=ij_participate_motif_number_list[edge_all.index((min(u,q),max(u,q)))]+1                
                deta1 += 1
                index0=edge_all.index((p,q))
                ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta1
        if len(v_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            for i in itertools.combinations(v_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    v_list0.append((min_pq,max_pq)) 
            deta2 = 0
            for p,q in v_list0:
                ij_participate_motif_number_list[edge_all.index((min(v,p),max(v,p)))]=ij_participate_motif_number_list[edge_all.index((min(v,p),max(v,p)))]+1
                ij_participate_motif_number_list[edge_all.index((min(v,q),max(v,q)))]=ij_participate_motif_number_list[edge_all.index((min(v,q),max(v,q)))]+1                
                deta2 += 1
                index0=edge_all.index((p,q))
                ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta2 
    return ij_participate_motif_number_list
def four_three_morphology_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n))    
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
        u_list0 = []
        v_list0 = []
        #如果节点u的邻居节点除v外只有一个，那么无法构成一个三角形，因此无模体
        if len(u_mor) <= 1:
            motif_matrix[u][v]=motif_matrix[u][v]+0
            motif_matrix[v][u]=motif_matrix[v][u]+0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(u_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    u_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in u_list0:
                motif_matrix[u][q]=motif_matrix[u][q]+1
                motif_matrix[q][u]=motif_matrix[q][u]+1
                motif_matrix[u][p]=motif_matrix[u][p]+1
                motif_matrix[p][u]=motif_matrix[p][u]+1
                
                motif_matrix[v][q]=motif_matrix[v][q]+1
                motif_matrix[q][v]=motif_matrix[q][v]+1
                motif_matrix[v][p]=motif_matrix[v][p]+1
                motif_matrix[p][v]=motif_matrix[p][v]+1               
                deta1 += 1
                motif_matrix[p][q]=motif_matrix[p][q]+1
                motif_matrix[q][p]=motif_matrix[q][p]+1   
            motif_matrix[u][v]=motif_matrix[u][v]+deta1
            motif_matrix[v][u]=motif_matrix[v][u]+deta1
        if len(v_mor) <= 1:
            motif_matrix[u][v]=motif_matrix[u][v]+0
            motif_matrix[v][u]=motif_matrix[v][u]+0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(v_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    v_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in v_list0:
                motif_matrix[v][q]=motif_matrix[v][q]+1
                motif_matrix[q][v]=motif_matrix[q][v]+1
                motif_matrix[v][p]=motif_matrix[v][p]+1
                motif_matrix[p][v]=motif_matrix[p][v]+1
                
                motif_matrix[u][q]=motif_matrix[u][q]+1
                motif_matrix[q][u]=motif_matrix[q][u]+1
                motif_matrix[u][p]=motif_matrix[u][p]+1
                motif_matrix[p][u]=motif_matrix[p][u]+1               
                deta1 += 1
                motif_matrix[p][q]=motif_matrix[p][q]+1
                motif_matrix[q][p]=motif_matrix[q][p]+1   
            motif_matrix[u][v]=motif_matrix[u][v]+deta1
            motif_matrix[v][u]=motif_matrix[v][u]+deta1
    return motif_matrix
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
                deta = int(len(cn_edge0))
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
def four_four_morphology_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
    for u,v in edge_all: 
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            motif_matrix[u][v]+=0
            motif_matrix[v][u]+=0
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                motif_matrix[u][v]+=0
                motif_matrix[v][u]+=0
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
                for p0,q0 in  cn_edge0: 
                   motif_matrix[p0][q0]+=1
                   motif_matrix[q0][p0]+=1     
                deta = int(len(cn_edge))
                motif_matrix[u][v]+=deta
                motif_matrix[v][u]+=deta
                #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
                node_list=[]
                node_number_list=[]
                for i in range(len(cn_edge0)):
                    for j in cn_edge0[i]:
                       node_list.append(j) 
                set_node_list=set(node_list)
                
                for i in set_node_list:
                    node_number_list.append([i,node_list.count(i)])
                print(node_number_list)
                #判断每个节点与u、v哪个节点相连接，并更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
                for i in range(len(node_number_list)):
                    node_i=node_number_list[i][0]
                    if (min(node_i,u),max(node_i,u)) in edge_all:
                        motif_matrix[u][node_i]+=node_number_list[i][1]
                        motif_matrix[node_i][u]+=node_number_list[i][1]    
                    if (min(node_i,v),max(node_i,v)) in edge_all:
                        motif_matrix[v][node_i]+=node_number_list[i][1]
                        motif_matrix[node_i][v]+=node_number_list[i][1]  
    return motif_matrix
#计算模体结构4-5(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def four_five_morphology(G,edge_all):
    ij_participate_motif_number_list=[]
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
        mor_list = []
        if (u_mor == []) or (v_mor == []):
            ij_participate_motif_number_list.append(0)
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list.append((i,j))
            deta = 0
            for p,q in mor_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta += 1
            ij_participate_motif_number_list.append(deta)
    return ij_participate_motif_number_list
def four_five_morphology_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
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
        mor_list = []
        if (u_mor == []) or (v_mor == []):
            pass
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list.append((i,j))
            deta = 0
            for p,q in mor_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    if p in u_friends:
                        motif_matrix[p][v]+=(1/4)
                        motif_matrix[v][p]+=(1/4)
                    else:
                        motif_matrix[p][u]+=(1/4)
                        motif_matrix[u][p]+=(1/4)   
                    if q in u_friends:
                        motif_matrix[q][v]+=(1/4)
                        motif_matrix[v][q]+=(1/4)
                    else:
                        motif_matrix[q][u]+=(1/4)
                        motif_matrix[u][q]+=(1/4)                     
                    deta += 1    
            motif_matrix[u][v]+=deta
            motif_matrix[v][u]+=deta
    return motif_matrix
##计算模体结构4-6(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/6为模体数量
def four_six_morphology(G,edge_all):
    ij_participate_motif_number_list=[]
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            ij_participate_motif_number_list.append(0)
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                ij_participate_motif_number_list.append(0)
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
                ij_participate_motif_number_list.append(deta)
    return ij_participate_motif_number_list   
#计算模体结构4-6(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/6为模体数量
def four_six_morphology_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            pass
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                pass
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
                motif_matrix[u][v]+=deta
                motif_matrix[v][u]+=deta
    return motif_matrix 

def float_rand(a, b, size):
   return a + ((b - a) * np.random.random(size))    

# 一行数组限定每个元素取值范围
def bound_SOSCD(l_bound, u_bound, x, n):
    for j in range(n):
        if x[j] < l_bound:
            x[j] = 2*l_bound - x[j]
            if x[j] > u_bound:
                x[j] = u_bound
        elif x[j] > u_bound:
            x[j] = 2*u_bound - x[j]
            if x[j] < l_bound:
                x[j] = l_bound            
    return x
def Wbias(X,Y,Gi,n,NP,weights,edge_all):
    #############################################################################
    # 确定权重最大的连边集合
    # 权重最大的连边序号
    edges_index_maxweight = [i for i,x in enumerate(weights) if x==max(weights)]
    # 权重最大的连边集合
    edges_maxweight = []
    for edge_index in edges_index_maxweight:
        edges_maxweight.append(edge_all[edge_index])
    #############################################################################
    # bias操作改进个体个数 
    improved_individual = 0
    # bias操作
    for i in range(NP):
        # 对于当前社区划分
        object_x = copy.deepcopy(X[i])
        #object_x = copy.deepcopy(best_x)
        #best_x = [4 5 5 4 6 4 8 1 1 1]  Qw = 0.06172
        # 针对每一条权重最大的连边
        for object_edge in edges_maxweight:
            # 最大权重边object_edge,对应的2个节点    
            object_v1 = object_edge[0]
            #print('object_v1 =',object_v1)
            object_v2 = object_edge[1]
            #print('object_v2 =',object_v2)
            #  2个节点的社区标号
            commid_v1 = object_x[object_v1]
            #print('commid_v1 =',commid_v1)
            commid_v2 = object_x[object_v2]
            #print('commid_v2 =',commid_v2)
            # 是否一致
            if commid_v1 != commid_v2:
                # 统一二者的社区标号
                uni_commid = random.choice([commid_v1,commid_v2]) 
                # 更新当前社区划分
                object_x[object_v1] = uni_commid
                object_x[object_v2] = uni_commid
                # array([5, 5, 5, 4, 6, 4, 8, 1, 1, 1])  Qw = 0.191358 
                # array([4, 4, 5, 4, 6, 4, 8, 1, 1, 1])  Qw = 0.16049
        # 计算bias操作后，object_x个体质量
        object_x_Qw = ig.GraphBase.modularity(Gi,object_x,weights)       
        # 若个体质量改进，更新种群
        if object_x_Qw > Y[i]:
            improved_individual += 1  
            X[i] = copy.deepcopy(object_x)
            Y[i] = copy.deepcopy(object_x_Qw)
    # 种群bias操作完毕
    return X,Y,improved_individual


#用于除社区修正外的QW优化
def moti_num(all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
#    GG = nx.from_pandas_dataframe(data,'src','dst',create_using=nx.Graph())
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    ij_participate_motif_number_list=three_one_morphology(GG,all_edge_G)       
    return ij_participate_motif_number_list    

#考虑模体邻居节点，用于社区修正
def moti_num_motifadd(all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
#    GG = nx.from_pandas_dataframe(data,'src','dst',create_using=nx.Graph())
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    matrix1=three_one_morphology_motifadd(GG,all_edge_G)       
    return matrix1  

network = "football"
print("########## {} ###########".format(network))
G = nx.read_edgelist(network +".txt")
G = G.to_undirected()
n=G.number_of_nodes()
#获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist(network +".txt")
Gi=Gi.subgraph(map(int,G.nodes()))          
Gi=Gi.as_undirected()
##################################################
edge_all = Gi.get_edgelist()
#基于模体加权列表，没有考虑模体邻居节点
ij_participate_motif_number_list = moti_num(edge_all)
############## 网络的权重  ###################
higher_weights = copy.deepcopy(ij_participate_motif_number_list)
degrees = Gi.degree()
ave_degree=sum(degrees)/n
lower_weights=[ave_degree/(sqrt(Gi.degree(i)*Gi.degree(j))) for i,j in edge_all]
################## 真实社区划分 ############################
df_news = pd.read_table(network +"_groundtruth.txt", header=None)
real_membership = []
for i in df_news[0]:
    real_membership.append(i)

#weights=[higher_weights[i]+lower_weights[i] for i in range(len(edge_all))]
#weights=copy.deepcopy(higher_weights)
###########################################################
#EMNCM操作：基于加权矩阵转化加权网络,用于求节点间最短路径
#模体邻接矩阵(考虑了基于模体的邻居节点,用于社区修正操作)
motif_matrix=moti_num_motifadd(edge_all)
#基于lower_weights生成低阶权重矩阵
lower_matrix=np.zeros((n,n))
for i in range(len(lower_weights)):
    lower_matrix[edge_all[i][0],edge_all[i][1]]=1#lower_weights[i]
    lower_matrix[edge_all[i][1],edge_all[i][0]]=1#lower_weights[i]
#高阶+低阶
matrix_lh=motif_matrix+lower_matrix
#基于path_weights生成加权网络，用于求节点间最短路径
new_edges=[]
for i in range(n):
    for j in range(n):
        if matrix_lh[i][j]>0:
            new_edges.append((i,j,1/matrix_lh[i][j]))
GG1=nx.Graph()
GG1.add_nodes_from(range(0,n))
for i in range(len(new_edges)):
    GG1.add_weighted_edges_from([(new_edges[i][0],new_edges[i][1],new_edges[i][2])])

###################提取将最短路径所有可能计算出来，降低时间复杂度##########################################
#matrix_path=np.zeros((n,n))
#for i in range(0,n):
#    for j in range(i,n):
#        if nx.has_path(GG1,i,j):            
#            matrix_path[i][j]=nx.dijkstra_path_length(GG1,i,j,weight= 'weight')
#            matrix_path[j][i]=nx.dijkstra_path_length(GG1,i,j,weight= 'weight')
#        else:
#            pass   
#############################################################
#EMNLS操作：基于higher_weights列表生成加权网络
motif_GG=nx.Graph()
motif_GG.add_nodes_from(range(0,n))
for i in range(len(edge_all)):
    motif_GG.add_weighted_edges_from([(edge_all[i][0],edge_all[i][1],higher_weights[i])])
############################################################
# 标准SOS算法参数设置
NP = 100  # The number of candidate solutions
Gen = 100000  # The number of iterations
# 网络参数设置
l_bound = 0
u_bound = n-1
#threshold_value=0.5
threshold_min=0.1
threshold_max=0.9
#######################
optimal_nmi = 1.0
optimal_Qw =  1.0
#######################

# 1. 构造初始种群
pop = np.random.randint(n, size=(NP, n))  # 100*10，0-99行，0-9列

# 计算适应度函数值
fit_Qw=[]
for i in range(NP):
    fit_Qw.append(ig.GraphBase.modularity(Gi,pop[i],higher_weights))
# 初始种群最优适应度值
best_fit_Qw = max(fit_Qw)
## 初始种群最优个体
best_x = pop[fit_Qw.index(max(fit_Qw))]

# 2.基于边的bias操作
# 统一每个社区划分中，最大权重连边的社区
Bias_pop=copy.deepcopy(pop)
Bias_fit_Qw=copy.deepcopy(fit_Qw)
# 种群中每个个体，进行基于权重的bias操作
##################################################################################################
[Bias_pop,Bias_fit_Qw,improved_individual] = Wbias(Bias_pop,Bias_fit_Qw,Gi,n,NP,higher_weights,edge_all)
##################################################################################################
#print('improved_individual =',improved_individual)     
#improved_individual = 93/86/90/99    

# bias操作之后，种群最优适应度值
best_Bias_fit_Qw = max(Bias_fit_Qw)
#print('best_Bias_fit_Qw =',best_Bias_fit_Qw) 
## bias操作之后，种群最优个体
best_Bias_x = Bias_pop[Bias_fit_Qw.index(max(Bias_fit_Qw))]
#print('best_Bias_x =',best_Bias_x)

# 3. SOSCD算法主循环,优化Qw
pop=copy.deepcopy(Bias_pop)
fit=copy.deepcopy(Bias_fit_Qw)    
# 初始化历史最优解
########################
best_nmi_history=[]
best_nmi_x_history=[]

best_fit_history=[]
best_fit_x_history=[]
best_fit_x_Q_history=[]
best_fit_x_nmi_history=[]
########################
#GRS-SOS参数设置
BFMIN=0.1
BFMAX=0.9
########################
exetime=1
p=0
##
while p<1:
    # 输出当前进化代数exetime
    
    # 3.1 Mutualism 
    Mutu_pop=copy.deepcopy(pop)
    Mutu_fit=copy.deepcopy(fit)
    for i in range(NP):
       # print(i)
       # (1)Xbest 
       best_fit_Mutu = max(Mutu_fit)
       Xbest = Mutu_pop[Mutu_fit.index(max(Mutu_fit))]
       # (2)Xj~=Xi
       j = np.random.permutation(np.delete(np.arange(NP), i))[0]
       # (3)Mutual Vector & Beneficial Factor
       # mutual_vector = sum(Mutu_pop)/NP
       # BF1 = BFMIN+(BFMAX-BFMIN)*(2-exp(exetime*log(2)/Gen))
       # BF2 = BFMIN+(BFMAX-BFMIN)*(exp(exetime*log(2)/Gen)-1)
       # #BF1, BF2 = np.random.randint(1, 3, 2)
       Xi,Xj = Mutu_pop[i],Mutu_pop[j]
       mutual_vector = 0.5 * (Xi + Xj) # 互利共生向量
       BF1=round(1+rd.random())
       BF2=round(1+rd.random())
       # 生成Xinew和Xjnew
       Xinew = Xi + rd.random()*(Xbest - BF1*mutual_vector)
       Xjnew = Xj + rd.random()*(Xbest - BF2*mutual_vector)
       # 限定取值范围
       Xinew = Xinew.round().astype(int) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       Xinew = bound_SOSCD(l_bound, u_bound, Xinew, n)
       # 限定取值范围
       Xjnew = Xjnew.round().astype(int) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       Xjnew = bound_SOSCD(l_bound, u_bound, Xjnew, n)
       # evaluate Xinew
       #Xinew_fit_Mutu = ig.GraphBase.modularity(Gi,Xinew_Mutu)
       Xinew_fit = ig.GraphBase.modularity(Gi,Xinew,higher_weights)
       Xjnew_fit = ig.GraphBase.modularity(Gi,Xjnew,higher_weights)

       # (5)updata Mutu_pop and Mutu_fit
       if Xinew_fit > Mutu_fit[i]:
           Mutu_pop[i] = Xinew
           Mutu_fit[i] = Xinew_fit
       if Xjnew_fit > Mutu_fit[j]:
           Mutu_pop[j] = Xjnew
           Mutu_fit[j] = Xjnew_fit
           
    # 3.2 Commensalism 
    Comm_pop=Mutu_pop
    Comm_fit=Mutu_fit
    for i in range(NP):
       # (1)Xbest 
       best_fit = max(Comm_fit)
       best_fit_index = Comm_fit.index(best_fit) 
       Xbest = [best_fit_index]
       # (2)Xj~=Xi
       # Xi != Xj != Xk
       ij_list = [i for i in range(NP)]
       ij_list.remove(i)
       j = rd.choice(ij_list)
       # 共栖算法
       Xi = Comm_pop[i]
       Xj = Comm_pop[j]
       Xinew = Xi + rd.uniform(-1, 1)*(Xbest - Xj)
       # 限定取值范围
       Xinew_Comm = Xinew.round().astype(int) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       Xinew_Comm = bound_SOSCD(l_bound, u_bound, Xinew_Comm, n)
       # evaluate Xinew 
       Xinew_fit_Comm = ig.GraphBase.modularity(Gi,Xinew_Comm,higher_weights)
       # (5)updata Comm_pop and Comm_fit
       if Xinew_fit_Comm > Comm_fit[i]:
           Comm_pop[i] = copy.deepcopy(Xinew_Comm)
           Comm_fit[i] = copy.deepcopy(Xinew_fit_Comm)

    # 3.3 Parasitism
    Para_pop=copy.deepcopy(Comm_pop)
    Para_fit=copy.deepcopy(Comm_fit)
    for i in range(NP):
       # 寄生算法
       para_vector = copy.deepcopy(Para_pop[i])   # 寄生向量
       seeds = [i for i in range(n)]
       rd.shuffle(seeds)
       pick = seeds[:rd.randint(1, n)] # 随机选择一些节点
       # 在约束范围内随机化节点的社区编号
       for ii in range(n): para_vector[ii] = rd.randint(1,n)    

       # 限定取值范围
       para_vector = para_vector.round().astype(int) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       para_vector = bound_SOSCD(l_bound, u_bound, para_vector, n)
       bestnew_x_Para_fit = ig.GraphBase.modularity(Gi,para_vector,higher_weights)
       if bestnew_x_Para_fit > Para_fit[i]:
           Para_pop[i] = para_vector
           Para_fit[i] = bestnew_x_Para_fit
           
    pop=copy.deepcopy(Para_pop)
    fit=copy.deepcopy(Para_fit)     
 

    # 当代种群最优Qw
    best_fit = max(fit) 
#    print('best_Qw=', best_fit) 
    best_fit_history.append(best_fit)   
    # 当代种群最优Qw对应个体
    best_fit_x = pop[fit.index(best_fit)]
    best_fit_x_history.append(best_fit_x)
    if exetime % 50 ==0:
        print ('exetime=',exetime)
        print('ceshibest_Qw =',best_fit) 
    #ig.GraphBase.modularity(Gi,best_fit_x,weights)
    # 当代种群最优个体,对应Q
    # best_fit_x_Q = ig.GraphBase.modularity(Gi,best_fit_x)
    # best_fit_x_Q_history.append(best_fit_x_Q)
    # print('best_fit_x_Q =',best_fit_x_Q)
    # 当代种群最优个体,对应nmi
#    best_fit_x_nmi = ig.compare_communities(real,best_fit_x,method='nmi',remove_none=False)
#    best_fit_x_nmi_history.append(best_fit_x_nmi)
#    print('best_fit_x_nmi =',best_fit_x_nmi)

             
    # 3.8 whether the loop stop?  
    # 人工合成网络
    #if (exetime>=Gen) or (abs(abs(best_x_with_nmi) - optimal_nmi) <= 1.0e-4):
    # 真实网络
    jishu = exetime-400
    
    if exetime>=Gen:
        p=1
    elif exetime >= 400 and best_fit_history[jishu] == best_fit_history[-1]:#30代不增长默认收敛
        print("最后迭代次数为",exetime-399)
        p = 1
    else:
       exetime+=1  


print('Qw',best_fit)
print("NMI",ig.compare_communities(real_membership,best_fit_x,method='nmi',remove_none=False))
        