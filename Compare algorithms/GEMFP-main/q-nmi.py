# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:38:16 2020

@author: Administrator
"""
# 标准SOS算法优化Q函数，进行社区检测
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

#计算模体结构3-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def m1(G,edge_all):
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
def m1_motifadd(G,edge_all):
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
def m2(G,edge_all):
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
def m2_motifadd(G,edge_all):
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
def m3(G,edge_all):    
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
def m3_motifadd(G,edge_all):
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
def m4(G,edge_all):
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
def m4_motifadd(G,edge_all):
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
def m5(G,edge_all):
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
def m5_motifadd(G,edge_all):
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
def m6(G,edge_all):
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
def m6_motifadd(G,edge_all):
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
                # print(node_number_list)
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
def m7(G,edge_all):
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
def m7_motifadd(G,edge_all):
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
def m8(G,edge_all):
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
def m8_motifadd(G,edge_all):
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
        
# def moti_num(motif_func_list,M,all_edge_G):    
#     data=DataFrame(list(all_edge_G),columns=['src','dst'])
#     GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    
#     #计算三阶模体 和M4(a,e,f)和S1和SS5
#     ij_participate_motif_number_list=[]
#     for i,j in all_edge_G:
#         ij_participate_motif_number=motif_func_list[M](i,j,GG)        
#         ij_participate_motif_number_list.append(ij_participate_motif_number)

def moti_num(motif_func_list, M, all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
#    GG = nx.from_pandas_dataframe(data,'src','dst',create_using=nx.Graph())
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    ij_participate_motif_number_list=motif_func_list[M](GG,all_edge_G)       
    return ij_participate_motif_number_list 
    #计算4-2,4-3,4-4 S2-S14 SS1-SS4模体 函数中遍历 
#    ij_participate_motif_number_list=four_two_morphology(GG,all_edge_G)       
    return ij_participate_motif_number_list
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

def clean_up_random(X,n,NP,threshold_value):
    # 对每个种群个体进行操作
    for i in range(NP):
        # 在个体i中随机选择get_num个节点进行社区调整
        get_num = np.random.randint(1, n)
        # 保存在use_node_index
        use_node_index = np.random.permutation(np.arange(n))[0:get_num]      
        # 对use_node_index中的节点,进行社区标号纠错
        for rand_i in range(get_num):           
           # 针对use_node_index中的每一个节点进行社区标号纠错
            node=use_node_index[rand_i]
            # 确定节点node的所有邻域个体
            neigh_node=Gi.neighbors(node)            
            # 构建节点node自身及邻居集合列表
            all_adj_node = copy.deepcopy(neigh_node)           
            all_adj_node.append(node)                  
            # node的社区编号
            node_comm = copy.deepcopy(X[i][node])
            # node的邻域节点的社区编号
            node_neigh_comm = copy.deepcopy(X[i][neigh_node]) 
            # 计算节点node的CV
            # 节点node与邻域个体属于不同社区的数目
            different_comm_number=0
            for k in node_neigh_comm:
                if node_comm!=k:
                   different_comm_number+=1
            # 节点node的度
            degree_node=len(neigh_node)
            # 节点node的CV值
            CV_node=float(different_comm_number)/degree_node
            # 判断CV是否大于阈值
            # 若是，则说明节点node与邻域节点不在同一社区的概率较大
            # 节点社区标号错误,选择邻域节点中出现的社区标号
            if CV_node >= threshold_value:
               # 邻域节点社区候选列表，及其规模 
               comm_list = copy.deepcopy(pd.value_counts(node_neigh_comm).index)
               comm_list = copy.deepcopy(comm_list.tolist())
               #print(comm_list)
               comm_num =  copy.deepcopy(pd.value_counts(node_neigh_comm).values)
               comm_num =  copy.deepcopy(comm_num.tolist())
               #根据comm_num确定comm_list中的最大规模社区和次大规模社区
               # 最大规模社区集合 = max_comm_id             
               max_comm_num = max(comm_num)
               max_comm_id = []
               max_comm_index = []
               for k in range(len(comm_num)):
                   if comm_num[k] == max_comm_num:
                       max_comm_id.append(comm_list[k])
                       max_comm_index.append(k)                       
               #从comm_num和comm_list中清除最大社区信息
               del_comm_num = []  #清除最大社区信息后的社区规模集合
               del_comm_list = [] #清除最大社区信息后的社区标号集合
               for k in range(len(comm_list)):
                   if comm_num[k] != max_comm_num:
                       del_comm_list.append(comm_list[k])
                       del_comm_num.append(comm_num[k])
               #次大规模社区集合 = second_max_comm_id 
               if len(del_comm_list) > 0:
                   second_max_comm_num = max(del_comm_num)
                   second_max_comm_id = []
                   for k in range(len(del_comm_num)):
                       if del_comm_num[k] == second_max_comm_num:
                           second_max_comm_id.append(del_comm_list[k])
               else:
                   second_max_comm_num = 0
                   second_max_comm_id = []
               # 最大规模社区集合 + 次大规模社区集合 = 可选社区集合
               comm_id_for_choice = copy.deepcopy(max_comm_id + second_max_comm_id)
#                逐个遍历可选社区集合comm_id_for_choice中的每个社区 
#                以不同的概率对节点node的社区进行重置
               for comm_id in comm_id_for_choice:                   
                   if comm_id == comm_id_for_choice[0] and (comm_id in max_comm_id):
                       #第1个最大规模社区，100%概率重置
                       X[i][node] = comm_id
                   elif (comm_id != comm_id_for_choice[0]) and (comm_id in max_comm_id):                       
                       if random.random() < 0.5:
                           X[i][node] = comm_id
                   elif (comm_id != comm_id_for_choice[0]) and (comm_id in second_max_comm_id):
                       if random.random() < 0.2:
                           X[i][node] = comm_id 
                
#               probability_number=random.random()
#               if (probability_number < 0.2) and (len(second_max_comm_id)!=0):
#                   comm_id=random.choice(second_max_comm_id)
#                   X[i][node] = comm_id
#               elif (probability_number >=0.2) and (len(max_comm_id)!=0):
#                   comm_id=random.choice(max_comm_id)
#                   X[i][node] = comm_id
                        
    # 一个种群个体操作完毕                         
    # 返回纠错后的新种群
    return X                   

def local_optimization(bestx,bestfit,n):
    # 对单个最优解个体进行局部搜索
    for i in range(n):       
        # 节点i的社区标号
        comm_id_i=bestx[i]
        # 节点i的邻居节点
        i_neighbors=Gi.neighbors(i)
        # 邻居节点的社区标号
        comm_id_i_neighbors=bestx[i_neighbors]
        # 节点i的备选社区标号
        comm_id_i_choice = copy.deepcopy(pd.value_counts(comm_id_i_neighbors).index)
        comm_id_i_choice = copy.deepcopy(comm_id_i_choice.tolist())
        if comm_id_i in comm_id_i_choice:
            comm_id_i_choice.remove(comm_id_i)           
        #逐个更换节点i的社区标号
        for j in range(len(comm_id_i_choice)):
            #print(j)
            x_change = copy.deepcopy(bestx)
            x_change[i] = comm_id_i_choice[j]
#            x_change_fit = ig.GraphBase.modularity(Gi,x_change)
            x_change_fit=ig.GraphBase.modularity(Gi,x_change,weights=ij_participate_motif_number_list)
            # 比较新个体与原个体的优劣
            if x_change_fit > bestfit:
                bestx = copy.deepcopy(x_change)
                bestfit = copy.deepcopy(x_change_fit)
    return bestx,bestfit     

# 网络信息
# =============================================================================
# 参数设置
# =============================================================================
network_name = "08" #网络
M = "M1"  #模体

G = nx.read_edgelist("./5000/network"+network_name+"_0.txt")
G = G.to_undirected()
n=G.number_of_nodes()
#获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist("./5000/network"+network_name+"_0.txt")
Gi=Gi.subgraph(map(int,G.nodes()))          
Gi=Gi.as_undirected()
##################################################
motif_func_list = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6, "M7": m7, "M8": m8}
edge_all = Gi.get_edgelist()
ij_participate_motif_number_list = moti_num(motif_func_list, M, edge_all)
#for i in range(len(ij_participate_motif_number_list)):
#    ij_participate_motif_number_list[i]=ij_participate_motif_number_list[i]+1
fo = pd.read_table(r"./5000/community"+network_name+"_0.txt",header = None)
real=[]
for i in fo[1]:
    real.append(i) 
# real=[]
# fo = open("../data/real/"+network_name+"_groundtruth.txt",'r')
# for line in fo:
#     real.append(int(float(line.split('\t')[1])))
# fo.close()
Qws,nmis = [],[]
for i in range(1):
    best_fit_x=[]
    fo = open("./5000/label"+network_name+"_0.txt",'r')
    for line in fo:
        best_fit_x.append(int(float(line)))
    fo.close()
    for i in range(len(best_fit_x)):
        ii=best_fit_x[i] 
        best_fit_x[i]=ii
    best_fit_x_nmi = ig.compare_communities(real,best_fit_x,method='nmi',remove_none=False) 
    best_x_Qw=ig.GraphBase.modularity(Gi,best_fit_x,weights=ij_participate_motif_number_list)
    Qws.append(best_x_Qw)
    nmis.append(best_fit_x_nmi)
print("###### {} ### {} ####".format(network_name,M))        
print('QW_mean={},std={}, max={}'.format(mean(Qws), std(Qws), max(Qws)))
print('NMI_mean={},std={}, max={}'.format(mean(nmis), std(nmis), max(nmis)))
