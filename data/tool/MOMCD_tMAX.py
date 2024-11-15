#	算法4：邻居社区总权重
#	Motif-SOSCDNW (优化Qw, WBias + clean_up_NW + local_search) 

#exetime = 10
#best_Qw = 0.4838408776166325 with Q = 0.41510519395134776 with nmi = 0.707135418720364
from igraph.clustering import VertexClustering
from geatpy import selecting
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
import matplotlib.pyplot as plt
from math import exp,log,sqrt 
from time import process_time
import heapq
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


def clean_up_NW(X,n,NP,threshold_value,weights,edge_all):
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
            neigh_nodes = Gi.neighbors(node)                             
            # node的社区编号
            node_comm = copy.deepcopy(X[i][node])
            # node邻域节点的社区编号,[]
            neigh_nodes_comm = copy.deepcopy(X[i][neigh_nodes])
            ####################    邻居社区的总权重     #############################
            # 邻域社区列表
            comm_list = copy.deepcopy(pd.value_counts(neigh_nodes_comm).index)
            comm_list = copy.deepcopy(comm_list.tolist())                        
            # 邻居社区的权重
            comm_weight = []
            # 每一个邻居社区的权重
            for k in range(len(comm_list)):
                # 第k个邻居社区的标号
                neighcomm = comm_list[k]
                # 属于当前neighcomm的所有邻居节点
                neighcomm_nodes_index = []
                neighcomm_nodes_index = [j for j,x in enumerate(neigh_nodes_comm) if x==neighcomm] 
                neighcomm_nodes = []               
                for index in neighcomm_nodes_index:
                    neighcomm_nodes.append(neigh_nodes[index])              
                # 属于当前neighcomm的每个邻居节点，对应边权重
                neighcomm_nodes_weight = []
                for neighcomm_node in neighcomm_nodes:
                    # edge = (node,neighcomm_node)
                    neighcomm_edge = (min(node,neighcomm_node),max(node,neighcomm_node))
                    # index(edge) 
                    neighcomm_edge_index = edge_all.index(neighcomm_edge)
                    # weight(edge)
                    neighcomm_nodes_weight.append(weights[neighcomm_edge_index])
                # 当前neighcomm的总权重
                comm_weight.append(np.sum(neighcomm_nodes_weight))
            ################################################################### 
            # 计算节点node的CV
            # 与节点node不同的社区，对应权重之和
            different_comm_weight=0             
            for k in range(len(comm_list)):
                if comm_list[k] != node_comm:
                   different_comm_weight += comm_weight[k]
            # 节点node所有邻域社区，对应平均权重之和
            all_comm_weight = np.sum(comm_weight)
            # 节点node的CV值
            CV_node=float(different_comm_weight)/all_comm_weight
            # 判断CV是否大于阈值
            # 若是，则说明节点node与其他邻域社区之间的权重较大
            # 节点社区标号错误
            if CV_node >= threshold_value:
##                逐个遍历可选社区集合comm_id_for_choice中的每个社区 
               #######################################################
               # 候选社区集合
               comm_id_for_choice = copy.deepcopy(comm_list)
               # 候选社区集合的平均权重 
               comm_id_for_choice_weight = copy.deepcopy(comm_weight)
               ########################################################
               # 如果候选社区只有1个，直接替换
               if len(comm_id_for_choice) == 1:                   
                   X[i][node] = comm_id_for_choice[0]
               # 如果候选社区有多个，按照轮盘赌法则选择
               else:
                   # 每个候选社区的概率
                   comm_id_for_choice_p = []                   
                   for k in range(len(comm_id_for_choice)):
                       # 第k个候选社区的概率
                       comm_id_for_choice_p.append(comm_id_for_choice_weight[k]/np.sum(comm_id_for_choice_weight)) 
                   # 每个候选社区的累加概率
                   comm_id_for_choice_interval = []
                   for k in range(len(comm_id_for_choice_p)):
                       # 第k个候选社区的累加概率
                       comm_id_for_choice_interval.append(sum(comm_id_for_choice_p[:k+1]))
                   # 轮盘赌概率选择
                   #从0到1之间选择一个随机数
                   random_p=random.random()
                   for k in range(len(comm_id_for_choice_interval)):
                       if k == 0 and random_p <= comm_id_for_choice_interval[k]:
                           comm_id = comm_id_for_choice[0]
                       elif k > 0 and  comm_id_for_choice_interval[k-1] < random_p <= comm_id_for_choice_interval[k]:
                           comm_id = comm_id_for_choice[k]
                   # 确定最终的社区标号        
                   X[i][node] = comm_id                     
    # 一个种群个体操作完毕                         
    # 返回纠错后的新种群
    return X       
#EMNCM操作
def EMNCM(X,n,NP,weights,edge_all,GG1):
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
#            print("node",node)
            # 确定节点node的所有邻域个体
            neigh_nodes = Gi.neighbors(node) 
            #有的网络有自环，会使得邻居社区是其自己
            neigh_nodes=list(set(neigh_nodes))#tianjia
            if node in neigh_nodes:#tianjia
                neigh_nodes.remove(node)#tianjia                            
            # node的社区编号
            node_comm = copy.deepcopy(X[i][node])
            # node邻域节点的社区编号,[]
            neigh_nodes_comm = copy.deepcopy(X[i][neigh_nodes])
            ####################    邻居社区的总权重     #############################
            # 邻域社区列表
            comm_list = copy.deepcopy(pd.value_counts(neigh_nodes_comm).index)
            comm_list = copy.deepcopy(comm_list.tolist())                        
            # 邻居社区权重
            comm_weight = []
            # 节点到邻居社区的最短路径
            allneighcomm_Shortest_path=[]
            # 每一个邻居社区的权重
            for k in range(len(comm_list)):
#                print("k",k)
                # 第k个邻居社区的标号
                neighcomm = comm_list[k]
#                print("neighcomm",neighcomm)
                #属于当前neighcomm的所有节点
                neighcommk_allnodes = [j for j,x in enumerate(X[i]) if x==neighcomm]
                #如果节点node在neighcomm内，那么neighcommk_allnodes去除节点node
                if node in neighcommk_allnodes:
                    neighcommk_allnodes.remove(node)
                #计算节点node到社区k内所有节点的最短路径,并求平均最短路径
                # 方式1 计算节点node到社区k内所有节点的最短路径,并求平均最短路径
#                neighcommk_Shortest_path=[nx.dijkstra_path_length(GG1,node,j,weight= 'weight') for j in neighcommk_allnodes]                
                #如果有些网络不是全联通网络，加判断语句
#                neighcommk_Shortest_path=[nx.dijkstra_path_length(GG1,node,j) for j in neighcommk_allnodes if nx.has_path(GG1,node,j)] 
                # 方式2 读取节点node到社区k内所有节点的最短路径,并求平均最短路径
                neighcommk_Shortest_path=[matrix_path[node][j] for j in neighcommk_allnodes if j!=0]#等于0的其实是不存在路径的，因此求节点到社区的节点的平均路径不应该考虑进去                
                allneighcomm_Shortest_path.append(np.mean(neighcommk_Shortest_path))#就是这行出的警告
                # 属于当前neighcomm的所有邻居节点
                neighcomm_nodes_index = []
                neighcomm_nodes_index = [j for j,x in enumerate(neigh_nodes_comm) if x==neighcomm] 
                neighcomm_nodes = []               
                for index in neighcomm_nodes_index:
                    neighcomm_nodes.append(neigh_nodes[index])              
                # 属于当前neighcomm的每个邻居节点，对应边权重
                neighcomm_nodes_weight = []
                for neighcomm_node in neighcomm_nodes:
                    # edge = (node,neighcomm_node)
                    neighcomm_edge = (min(node,neighcomm_node),max(node,neighcomm_node))
                    # index(edge) 
                    neighcomm_edge_index = edge_all.index(neighcomm_edge)
                    # weight(edge)
                    neighcomm_nodes_weight.append(weights[neighcomm_edge_index])
#                print("neighcomm_nodes_weight",neighcomm_nodes_weight)
                # 当前neighcomm的总权重
                comm_weight.append(np.sum(neighcomm_nodes_weight))
            #每个邻居社区对节点node的吸引力
            attr_node_neighcomm=[comm_weight[i]/allneighcomm_Shortest_path[i] for i in range(len(allneighcomm_Shortest_path))]    
            #节点node对每个邻居社区的隶属度  
            attr_node_neighcomm_sum=np.sum(attr_node_neighcomm)
            belongdegree_node_neighcomm=[i/attr_node_neighcomm_sum for i in attr_node_neighcomm] 
            avg_belongdegree_node_neighcomm=np.mean(belongdegree_node_neighcomm)
#            print("attr_node_neighcomm",attr_node_neighcomm)
            ################################################################### 
            #判断节点i是否跟邻居属于一个社区，如果是，那么判断当前社区隶属度是否大于均值，若小于，社区划分不合理
            #判断节点i是否跟邻居属于一个社区,如果，社区划分不合理
            if node_comm not in comm_list or belongdegree_node_neighcomm[comm_list.index(node_comm)]<avg_belongdegree_node_neighcomm:
               # 候选社区集合
               comm_id_for_choice = copy.deepcopy(comm_list)
               # 候选社区有多个，按照轮盘赌法则选择
               # 每个候选社区的累加概率
               comm_id_for_choice_interval = []
               for k in range(len(belongdegree_node_neighcomm)):
                   # 第k个候选社区的累加概率
                   comm_id_for_choice_interval.append(sum(belongdegree_node_neighcomm[:k+1]))
               # 轮盘赌概率选择
               #从0到1之间选择一个随机数
               random_p=random.random()
               for k in range(len(comm_id_for_choice_interval)):
                   if k == 0 and random_p <= comm_id_for_choice_interval[k]:
                       comm_id = comm_id_for_choice[0]
                   elif k > 0 and  comm_id_for_choice_interval[k-1] < random_p <= comm_id_for_choice_interval[k]:
                       comm_id = comm_id_for_choice[k]
               # 确定最终的社区标号        
               X[i][node] = comm_id  
#            print("X",X)                   
    # 一个种群个体操作完毕                         
    # 返回纠错后的新种群
    return X       
#EMNCM操作，针对于m2等非全联通模体我们要考虑模体邻居社区，模体邻居节点
def EMNCM_addMotifneighborhood(X,n,NP,edge_all,GG1,motif_matrix,lower_weights):
############变化##########################
    #基于lower_weights生成低阶权重矩阵
#    lower_matrix=np.zeros((n,n))
#    for i in range(len(lower_weights)):
#        lower_matrix[edge_all[i][0],edge_all[i][1]]=lower_weights[i]
#        lower_matrix[edge_all[i][1],edge_all[i][0]]=lower_weights[i]
    matrix_lh=motif_matrix+lower_matrix
########################################   
########################################             
    # 对每个种群个体进行操作
    for i in range(NP):
        # 在个体i中随机选择get_num个节点进行社区调整
        get_num = int(1.0*n)#np.random.randint(1, n)
        # 保存在use_node_index
        use_node_index = np.random.permutation(np.arange(n))[0:get_num]      
        # 对use_node_index中的节点,进行社区标号纠错
        for rand_i in range(get_num):           
           # 针对use_node_index中的每一个节点进行社区标号纠错
            node=use_node_index[rand_i]
#            print("node",node)
#            # 确定节点node的所有邻域个体
#            neigh_nodes = Gi.neighbors(node)  
##################变化###########################################
            # 确定节点node的所有邻域节点      
            # 确定节点node的所有邻域节点(包括节点i的边邻居节点和模体邻居节点)
            neigh_nodes=np.where(matrix_lh[node]>0)
            neigh_nodes=list(neigh_nodes[0][:])  
################################################################                             
            # node的社区编号
            node_comm = copy.deepcopy(X[i][node])
            # node邻域节点的社区编号,[]
            neigh_nodes_comm = copy.deepcopy(X[i][neigh_nodes])
          
            ####################    邻居社区的总权重     #############################
            # 邻域社区列表
            comm_list = copy.deepcopy(pd.value_counts(neigh_nodes_comm).index)
            comm_list = copy.deepcopy(comm_list.tolist())                        
            # 邻居社区权重
            comm_weight = []
            # 节点到邻居社区的最短路径
#            allneighcomm_Shortest_path=[]
            # 每一个邻居社区的权重
            for k in range(len(comm_list)):
#                print("k",k)
                # 第k个邻居社区的标号
                neighcomm = comm_list[k]
#                print("neighcomm",neighcomm)
                #属于当前neighcomm的所有节点
                neighcommk_allnodes = [j for j,x in enumerate(X[i]) if x==neighcomm]
                #如果节点node在neighcomm内，那么neighcommk_allnodes去除节点node
                if node in neighcommk_allnodes:
                    neighcommk_allnodes.remove(node)
                # 方式1 计算节点node到社区k内所有节点的最短路径,并求平均最短路径
#                neighcommk_Shortest_path=[nx.dijkstra_path_length(GG1,node,j,weight= 'weight') for j in neighcommk_allnodes]                
                #如果有些网络不是全联通网络，加判断语句
#                neighcommk_Shortest_path=[nx.dijkstra_path_length(GG1,node,j,weight= 'weight') for j in neighcommk_allnodes if nx.has_path(GG1,node,j)] 
                # 方式2 计算读取节点node到社区k内所有节点的最短路径,并求平均最短路径
#                neighcommk_Shortest_path=[matrix_path[node][j] for j in neighcommk_allnodes if j!=0]#等于0的其实是不存在路径的，因此求节点到社区的节点的平均路径不应该考虑进去
#                allneighcomm_Shortest_path.append(np.mean(neighcommk_Shortest_path))
#                print("neighcommk_Shortest_path",neighcommk_Shortest_path)
                
                # 属于当前neighcomm的所有邻居节点
                neighcomm_nodes_index = []
                neighcomm_nodes_index = [j for j,x in enumerate(neigh_nodes_comm) if x==neighcomm] 
                neighcomm_nodes = []               
                for index in neighcomm_nodes_index:
                    neighcomm_nodes.append(neigh_nodes[index])              
                # 属于当前neighcomm的每个邻居节点，对应边权重
                neighcomm_nodes_weight = []
                for neighcomm_node in neighcomm_nodes:
                    # edge = (node,neighcomm_node)
#                    neighcomm_edge = (min(node,neighcomm_node),max(node,neighcomm_node))
#                    # index(edge) 
#                    neighcomm_edge_index = edge_all.index(neighcomm_edge)
#                    # weight(edge)
#                    neighcomm_nodes_weight.append(weights[neighcomm_edge_index])
####################变化##########################################
                    neighcomm_nodes_weight.append(matrix_lh[node,neighcomm_node])
##############################################################
#                print("neighcomm_nodes_weight",neighcomm_nodes_weight)
                # 当前neighcomm的总权重
                comm_weight.append(np.sum(neighcomm_nodes_weight))
            #每个邻居社区对节点node的吸引力
#            attr_node_neighcomm=[comm_weight[i]/allneighcomm_Shortest_path[i] for i in range(len(allneighcomm_Shortest_path))]    
            #不考虑最短路径
            attr_node_neighcomm=copy.deepcopy(comm_weight)    
            #节点node对每个邻居社区的隶属度  
            attr_node_neighcomm_sum=np.sum(attr_node_neighcomm)
            belongdegree_node_neighcomm=[i/attr_node_neighcomm_sum for i in attr_node_neighcomm] 
            avg_belongdegree_node_neighcomm=np.mean(belongdegree_node_neighcomm)
#            print("attr_node_neighcomm",attr_node_neighcomm)
#            print("belongdegree_node_neighcomm",belongdegree_node_neighcomm)
            ################################################################### 
            #判断节点i是否跟邻居属于一个社区，如果是，那么判断当前社区隶属度是否大于均值，若小于，社区划分不合理
            #判断节点i是否跟邻居属于一个社区,如果，社区划分不合理
            if node_comm not in comm_list or belongdegree_node_neighcomm[comm_list.index(node_comm)]<=0.7:
               # 候选社区集合
               comm_id_for_choice = copy.deepcopy(comm_list)
               # 候选社区有多个，按照轮盘赌法则选择
               # 每个候选社区的累加概率
               comm_id_for_choice_interval = []
               for k in range(len(belongdegree_node_neighcomm)):
                   # 第k个候选社区的累加概率
                   comm_id_for_choice_interval.append(sum(belongdegree_node_neighcomm[:k+1]))
               # 轮盘赌概率选择
               #从0到1之间选择一个随机数
               random_p=random.random()
               for k in range(len(comm_id_for_choice_interval)):
                   if k == 0 and random_p <= comm_id_for_choice_interval[k]:
                       comm_id = comm_id_for_choice[0]
                   elif k > 0 and  comm_id_for_choice_interval[k-1] < random_p <= comm_id_for_choice_interval[k]:
                       comm_id = comm_id_for_choice[k]
               # 确定最终的社区标号        
               X[i][node] = comm_id  
#            print("X",X)                   
    # 一个种群个体操作完毕                         
    # 返回纠错后的新种群
    return X      

def EMNCM_addMotifneighborhood2(X,n,NP,edge_all,GG1,motif_matrix,lower_weights):
############变化##########################
    #基于lower_weights生成低阶权重矩阵
#    lower_matrix=np.zeros((n,n))
#    for i in range(len(lower_weights)):
#        lower_matrix[edge_all[i][0],edge_all[i][1]]=lower_weights[i]
#        lower_matrix[edge_all[i][1],edge_all[i][0]]=lower_weights[i]
    matrix_lh=motif_matrix+lower_matrix
########################################   
########################################             
    # 对每个种群个体进行操作
    for i in range(NP):
        # 在个体i中随机选择get_num个节点进行社区调整
        get_num = int(1.0*n)#np.random.randint(1, n)
        # 保存在use_node_index
        use_node_index = np.random.permutation(np.arange(n))[0:get_num]      
        cluster_list = list(VertexClustering(Gi,X[i]))
        # 对use_node_index中的节点,进行社区标号纠错
        for rand_i in range(get_num):           
           # 针对use_node_index中的每一个节点进行社区标号纠错
            node=use_node_index[rand_i]
            # print(node)
##################变化###########################################
            # 确定节点node的所有邻域节点      
            # 确定节点node的所有邻域节点(包括节点i的边邻居节点和模体邻居节点)
            neigh_nodes=np.where(matrix_lh[node]>0)
            neigh_nodes=list(neigh_nodes[0][:])  
################################################################                             
            node_comm = copy.deepcopy(X[i][node])
            neigh_nodes_comm = copy.deepcopy(X[i][neigh_nodes])
            ####################    邻居社区的总权重     #############################
            # 邻域社区列表
            comm_list = copy.deepcopy(pd.value_counts(neigh_nodes_comm).index)
            comm_list = copy.deepcopy(comm_list.tolist())                        
            # 邻居社区权重
            comm_weight = []
            # 每一个邻居社区的权重
            for k in range(len(comm_list)):
#                print("k",k)
                # 第k个邻居社区的标号
                neighcomm = comm_list[k]
                
                # 属于当前neighcomm的所有邻居节点

                
                neighcomm_nodes = list(set(cluster_list[neighcomm]) & set(neigh_nodes))
                
                # print(neighcomm_nodes)

                # # neighcomm_nodes_index = []
                # neighcomm_nodes_index = [j for j,x in enumerate(neigh_nodes_comm) if x==neighcomm]                 
                # neighcomm_nodes = []               
                # for index in neighcomm_nodes_index:
                #     neighcomm_nodes.append(neigh_nodes[index])  
                # print(set(neighcomm_nodes))
                # xxx
                # 属于当前neighcomm的每个邻居节点，对应边权重
                neighcomm_nodes_weight = []
                for neighcomm_node in neighcomm_nodes:
                    
                    neighcomm_nodes_weight.append(matrix_lh[node,neighcomm_node])
##############################################################
#                print("neighcomm_nodes_weight",neighcomm_nodes_weight)
                # 当前neighcomm的总权重
                comm_weight.append(np.sum(neighcomm_nodes_weight))
            #每个邻居社区对节点node的吸引力
#            attr_node_neighcomm=[comm_weight[i]/allneighcomm_Shortest_path[i] for i in range(len(allneighcomm_Shortest_path))]    
            #不考虑最短路径
            attr_node_neighcomm=copy.deepcopy(comm_weight)    
            #节点node对每个邻居社区的隶属度  
            attr_node_neighcomm_sum=np.sum(attr_node_neighcomm)
            # belongdegree_node_neighcomm=[j/attr_node_neighcomm_sum for j in attr_node_neighcomm] 
            belongdegree_node_neighcomm=attr_node_neighcomm/attr_node_neighcomm_sum
            if node_comm not in comm_list or belongdegree_node_neighcomm[comm_list.index(node_comm)] <= 0.7:
                
                list2 = (np.array(belongdegree_node_neighcomm) / sum(belongdegree_node_neighcomm)).reshape(-1,1)
                comm_id_index = selecting('rws', list2, 1)[0]#锦标赛
                # comm_id_index = np.argmax(list2)
                comm_id = comm_list[comm_id_index]
                
                # comm_id = np.argmax(list1)
                X[i][node] = comm_id
                cluster_list[node_comm].remove(node)
                cluster_list[comm_id].append(node)
            
#             avg_belongdegree_node_neighcomm=np.mean(belongdegree_node_neighcomm)
# #            print("attr_node_neighcomm",attr_node_neighcomm)
# #            print("belongdegree_node_neighcomm",belongdegree_node_neighcomm)
#             ################################################################### 
#             #判断节点i是否跟邻居属于一个社区，如果是，那么判断当前社区隶属度是否大于均值，若小于，社区划分不合理
#             #判断节点i是否跟邻居属于一个社区,如果，社区划分不合理
#             if node_comm not in comm_list or belongdegree_node_neighcomm[comm_list.index(node_comm)]<=0.7:
#                # 候选社区集合
#                comm_id_for_choice = copy.deepcopy(comm_list)
#                # 候选社区有多个，按照轮盘赌法则选择
#                # 每个候选社区的累加概率
#                comm_id_for_choice_interval = []
#                for k in range(len(belongdegree_node_neighcomm)):
#                    # 第k个候选社区的累加概率
#                    comm_id_for_choice_interval.append(sum(belongdegree_node_neighcomm[:k+1]))
#                # 轮盘赌概率选择
#                #从0到1之间选择一个随机数
#                random_p=random.random()
#                for k in range(len(comm_id_for_choice_interval)):
#                    if k == 0 and random_p <= comm_id_for_choice_interval[k]:
#                        comm_id = comm_id_for_choice[0]
#                    elif k > 0 and  comm_id_for_choice_interval[k-1] < random_p <= comm_id_for_choice_interval[k]:
#                        comm_id = comm_id_for_choice[k]
#                # 确定最终的社区标号        
#                X[i][node] = comm_id  


#            print("X",X)                   
    # 一个种群个体操作完毕                         
    # 返回纠错后的新种群
    return X     

def differ_Qw(X,move_node,oldcomm_id,newcomm_id,higher_weights,edge_all,comm_id_i_neighbors,i_neighbors,motif_GG):
    #查看新社区newcomm_id中的邻居节点
    newcomm_neighnodes_index = [j for j,x in enumerate(comm_id_i_neighbors) if x==newcomm_id]                
    newcomm_neighnodes=[i_neighbors[index] for index in newcomm_neighnodes_index]
    innernewcomm_changeweight=[higher_weights[edge_all.index((min(move_node,node_j),max(move_node,node_j)))] for node_j in newcomm_neighnodes]
    innernewcomm_changeweight_sum=sum(innernewcomm_changeweight)
    #查看旧社区的邻居节点
    oldcomm_neighnodes_index = [j for j,x in enumerate(comm_id_i_neighbors) if x==oldcomm_id]                
    oldcomm_neighnodes=[i_neighbors[index] for index in oldcomm_neighnodes_index] 
    #旧社团内部边的权值总和变化
    inneroldcomm_changeweight=[higher_weights[edge_all.index((min(move_node,node_j),max(move_node,node_j)))] for node_j in oldcomm_neighnodes]
    inneroldcomm_changeweight_sum=sum(inneroldcomm_changeweight)
    #新社团中所有节点的强度之和
    newcomm_nodes=[j for j,x in enumerate(X) if x==newcomm_id]
    newcommnodes_strength=[motif_GG.degree(i,weight='weight') for i in newcomm_nodes]
    newcommnodes_strength_sum=sum(newcommnodes_strength)    
    #旧社团中所有节点的强度之和
    oldcomm_nodes=[j for j,x in enumerate(X) if x==oldcomm_id]
    oldcommnodes_strength=[motif_GG.degree(i,weight='weight') for i in oldcomm_nodes]
    oldcommnodes_strength_sum=sum(oldcommnodes_strength)    
    #迁徙节点i的节点强度
    move_node_strength=motif_GG.degree(move_node,weight='weight')
    #网络总权重
    alledges_weights_sum=sum(higher_weights)
    differ_Qw=(1/(2*alledges_weights_sum))*(2*(innernewcomm_changeweight_sum-inneroldcomm_changeweight_sum)-(move_node_strength*(move_node_strength+newcommnodes_strength_sum-oldcommnodes_strength_sum)/alledges_weights_sum))
    return differ_Qw
def EMNLS(Gi,bestx,bestfit,n,higher_weights,edge_all,motif_GG):
    # 对单个最优解个体进行局部搜索
    for i in range(n):  
        #选择其中社区归属与邻居节点不完全一致的边界节点，进行邻居社区局部搜索
        if set(bestx[Gi.neighbors(i)])!= {bestx[i]}: 
            # 节点i的社区标号
            comm_id_i=bestx[i]
            # 节点i的邻居节点
            i_neighbors=Gi.neighbors(i)
#            i_neighbors=np.where(matrix_lh[i]>0)
            #有的网络有自环，会使得邻居社区是其自己
            i_neighbors=list(set(i_neighbors))#tianjia
            if i in i_neighbors:#tianjia
                i_neighbors.remove(i)#tianjia             
            # 邻居节点的社区标号
            comm_id_i_neighbors=bestx[i_neighbors]
            # 节点i的备选社区标号
            comm_id_i_choice = copy.deepcopy(pd.value_counts(comm_id_i_neighbors).index)
            comm_id_i_choice = copy.deepcopy(comm_id_i_choice.tolist())
            if comm_id_i in comm_id_i_choice:
                comm_id_i_choice.remove(comm_id_i)           
            #逐个更换节点i的社区标号
            #迁移到每个邻居社区的增量Q列表
            x_change_diff_fit_list=[]
            for j in range(len(comm_id_i_choice)):                
                x_change_diff_fit=differ_Qw(bestx,i,comm_id_i,comm_id_i_choice[j],higher_weights,edge_all,comm_id_i_neighbors,i_neighbors,motif_GG)
                x_change_diff_fit_list.append(x_change_diff_fit)
            # 比较新个体与原个体的优劣
            if max(x_change_diff_fit_list) > 0:
                x_change_diff_fit_list_index=x_change_diff_fit_list.index(max(x_change_diff_fit_list))
                bestx[i] = comm_id_i_choice[x_change_diff_fit_list_index]
                bestfit = bestfit+max(x_change_diff_fit_list)    
    return bestx,bestfit 
def local_optimization(bestx,bestfit,n):
    # 对单个最优解个体进行局部搜索
    for i in range(n):       
        # 节点i的社区标号
        comm_id_i=bestx[i]
        # 节点i的邻居节点
#        i_neighbors=Gi.neighbors(i)
        i_neighbors=np.where(matrix_lh[i]>0)
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
            #
            x_change_fit=ig.GraphBase.modularity(Gi,x_change,higher_weights)
            # 比较新个体与原个体的优劣
            if x_change_fit > bestfit:
                bestx = copy.deepcopy(x_change)
                bestfit = copy.deepcopy(x_change_fit)
    return bestx,bestfit 
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
############################################################################################################
#real=[]
#fo = open("brain47_groundtruth.txt",'r')
#for line in fo:
#    real.append(int(float(line)))
#fo.close()
#real=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
# 网络信息
G = nx.read_edgelist("netscience_lianxu_wu.txt")
G = G.to_undirected()
n=G.number_of_nodes()
#获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist("netscience_lianxu_wu.txt")
Gi=Gi.subgraph(map(int,G.nodes()))          
Gi=Gi.as_undirected()
##################################################
edge_all = Gi.get_edgelist()
#基于模体加权列表，没有考虑模体邻居节点
ij_participate_motif_number_list = moti_num(edge_all)
############## 网络的权重  ###################
higher_weights = copy.deepcopy(ij_participate_motif_number_list)
# xxx
degrees = Gi.degree()
ave_degree=sum(degrees)/n
lower_weights=[ave_degree/(sqrt(Gi.degree(i)*Gi.degree(j))) for i,j in edge_all]
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

# st1 = process_time()

# 标准SOS算法参数设置
NP = 100  # The number of candidate solutions
Gen = 100  # The number of iterations
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
# st2 = process_time()
##
while p<1:
    # 输出当前进化代数exetime
    print ('exetime=',exetime)
    
    # 3.1 Mutualism 
    Mutu_pop=copy.deepcopy(pop)
    Mutu_fit=copy.deepcopy(fit)
    for i in range(NP):
       # print(i)
       # (1)Xbest 
       best_fit_Mutu = max(Mutu_fit)
       best_x_Mutu = copy.deepcopy(Mutu_pop[Mutu_fit.index(max(Mutu_fit))])
       # (2)Xj~=Xi
       j = np.random.permutation(np.delete(np.arange(NP), i))[0]
       # (3)Mutual Vector & Beneficial Factor
       mutual_vector = sum(Mutu_pop)/NP
       BF1 = BFMIN+(BFMAX-BFMIN)*(2-exp(exetime*log(2)/Gen))
       BF2 = BFMIN+(BFMAX-BFMIN)*(exp(exetime*log(2)/Gen)-1)
       #BF1, BF2 = np.random.randint(1, 3, 2)
       # (4)Xinew
       Xinew_Mutu = copy.deepcopy(Mutu_pop[j] + BF1 * (best_x_Mutu - Mutu_pop[i]) + BF2 * (mutual_vector - Mutu_pop[i]))
       # 限定取值范围
       Xinew_Mutu = copy.deepcopy(Xinew_Mutu.round().astype(int)) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       Xinew_Mutu = copy.deepcopy(bound_SOSCD(l_bound, u_bound, Xinew_Mutu, n))
       # evaluate Xinew
       #Xinew_fit_Mutu = ig.GraphBase.modularity(Gi,Xinew_Mutu)
       Xinew_fit_Mutu = ig.GraphBase.modularity(Gi,Xinew_Mutu,higher_weights)
       # (5)updata Mutu_pop and Mutu_fit
       if Xinew_fit_Mutu > Mutu_fit[i]:
           Mutu_pop[i] = copy.deepcopy(Xinew_Mutu)
           Mutu_fit[i] = copy.deepcopy(Xinew_fit_Mutu)           
           
    # 3.2 Commensalism 
    Comm_pop=copy.deepcopy(Mutu_pop)
    Comm_fit=copy.deepcopy(Mutu_fit)
    for i in range(NP):
       # (1)Xbest 
       best_fit_Comm = max(Comm_fit)
       bad_fit_Comm = min(Comm_fit)
       best_x_Comm = copy.deepcopy(Comm_pop[Comm_fit.index(max(Comm_fit))])
       # (2)Xj~=Xi
       random_three=np.random.permutation(np.delete(np.arange(NP), i))
       j_first = random_three[0]
       j_second = random_three[1]
       j_third = random_three[2]
       Q1=np.percentile(Comm_fit,25)
       Q3=np.percentile(Comm_fit,75)
       # (3)Xinew 
       array_rand = (Comm_fit[i]-bad_fit_Comm)/(max(Comm_fit)-bad_fit_Comm + 0.000001)
       if Comm_fit[i]>=Q3:           
           Xinew_Comm = copy.deepcopy(Comm_pop[i] + array_rand * (Comm_pop[j_second] - Comm_pop[j_third]))       
       elif Comm_fit[i]<=Q1:
           Xinew_Comm = copy.deepcopy(Comm_pop[j_first] + array_rand * (best_x_Comm - Comm_pop[i]))
       else:
           Xinew_Comm = copy.deepcopy(Comm_pop[i] + array_rand * (best_x_Comm - Comm_pop[j_first]))
       # 限定取值范围
       Xinew_Comm = copy.deepcopy(Xinew_Comm.round().astype(int)) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       Xinew_Comm = copy.deepcopy(bound_SOSCD(l_bound, u_bound, Xinew_Comm, n))
       # evaluate Xinew 
       #Xinew_fit_Comm = ig.GraphBase.modularity(Gi,Xinew_Comm)
       Xinew_fit_Comm = ig.GraphBase.modularity(Gi,Xinew_Comm,higher_weights)
       # (5)updata Comm_pop and Comm_fit
       if Xinew_fit_Comm > Comm_fit[i]:
           Comm_pop[i] = copy.deepcopy(Xinew_Comm)
           Comm_fit[i] = copy.deepcopy(Xinew_fit_Comm)

    # 3.3 Parasitism
    Para_pop=copy.deepcopy(Comm_pop)
    Para_fit=copy.deepcopy(Comm_fit)
    for i in range(NP):
       best_fit_Para = max(Para_fit)
       best_x_Para_index=Para_fit.index(max(Para_fit))
       best_x_Para = copy.deepcopy(Para_pop[best_x_Para_index])
       # (1) Xj~=Xi
       j = np.random.permutation(np.delete(np.arange(NP), i))[0]
       # (2) Parasite Vector 
       Para_vector=copy.deepcopy(Para_pop[i])
       # 方式1：仅随机更改一维分量
       # Para_vector[np.random.randint(0, dim)] = float_rand(l_bound, u_bound,1)
       # 方式2：随机更改多维分量 
       change_number = np.random.randint(1, n)
       pick = np.random.permutation(np.arange(n))[0:change_number]
       for index in pick:
           Para_vector[index] = np.random.randint(n)      
       # 限定取值范围
       Para_vector = copy.deepcopy(Para_vector.round().astype(int)) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       Para_vector = copy.deepcopy(bound_SOSCD(l_bound, u_bound, Para_vector, n))        
       # (3) evaluate Parasite Vector 
       Para_vector_fit = ig.GraphBase.modularity(Gi,Para_vector,higher_weights)
       # (4) evaluate IF
       IF=(Para_vector_fit-min(Para_fit))/(max(Para_fit)-min(Para_fit)+0.000001)
#       IF=(max(Para_fit)-Para_vector_fit)/(max(Para_fit)-min(Para_fit)+0.000001)
       bestnew_x_Para= copy.deepcopy(best_x_Para+IF*(Para_vector-best_x_Para))
       # 限定取值范围
       bestnew_x_Para = copy.deepcopy(bestnew_x_Para.round().astype(int)) # 取整，变为离散整数
       # 每一维元素限定至[0,n)之间
       bestnew_x_Para = copy.deepcopy(bound_SOSCD(l_bound, u_bound, bestnew_x_Para, n))
       bestnew_x_Para_fit = ig.GraphBase.modularity(Gi,bestnew_x_Para,higher_weights)
       if bestnew_x_Para_fit > best_fit_Para:
           Para_pop[best_x_Para_index] = copy.deepcopy(bestnew_x_Para)
           Para_fit[best_x_Para_index] = copy.deepcopy(bestnew_x_Para_fit)
           
    # 3.4 clean_up_NW
    Para_pop_copy = copy.deepcopy(Para_pop)
    # 对Para_pop_copy中每个个体进行节点修正
    ###################################################################################################
#    Para_clean_pop = copy.deepcopy(clean_up_NW_new(Para_pop_copy,n,NP,threshold_value,edge_all,motif_matrix))   
#    Para_clean_pop = copy.deepcopy(clean_up_NW(Para_pop_copy,n,NP,threshold_value,weights,edge_all))
#    Para_clean_pop = copy.deepcopy(EMNCM(Para_pop_copy,n,NP,weights,edge_all,GG1))
    # st11 = process_time()
    # Para_clean_pop = copy.deepcopy(EMNCM_addMotifneighborhood(Para_pop_copy,n,NP,edge_all,GG1,motif_matrix,lower_weights))
    # st22 = process_time()
    Para_clean_pop = copy.deepcopy(EMNCM_addMotifneighborhood2(Para_pop_copy,n,NP,edge_all,GG1,motif_matrix,lower_weights))
    # st33 = process_time()
    
    # print(st22-st11)
    # print(st33-st22)
    # xxxx
    ####################################################################################################
    
    # 计算适应度值
    Para_clean_fit = []
    for i in range(NP):        
        #############################################################################        
        Para_clean_fit.append(ig.GraphBase.modularity(Gi,Para_clean_pop[i],higher_weights))
        #############################################################################
    # 根据Para_clean_pop，更新 Para_pop 和 Para_fit
    # better_number=0
    for i in range(NP):
        # 统计上述操作中，改进个体数目
        # if Para_clean_fit[i] > Para_fit[i]:
        #     better_number = better_number+1
        # 即使Qw值是保持不变的，也保留节点修正结果，使其与邻域节点在一个社区，提高NMI
        # >=
        if Para_clean_fit[i] >= Para_fit[i]:
            Para_pop[i] = copy.deepcopy(Para_clean_pop[i])
            Para_fit[i] = copy.deepcopy(Para_clean_fit[i])            

    # 3.5 局部搜索
    ls_pop = copy.deepcopy(Para_pop)
    ls_fit = copy.deepcopy(Para_fit)
    # better_number=0
    mean_fit = (heapq.nlargest(int(NP*0.2), ls_fit))[-1] #求最大的三个元素，并排序
    for i in range(NP):
        if ls_fit[i] > mean_fit:
            ###############################################################
#            [temp_x,temp_fit] = EMNLS(Gi,ls_pop[i],ls_fit[i],n,higher_weights,edge_all,motif_GG)
            ###############################################################            
            [temp_x,temp_fit] = local_optimization(ls_pop[i],ls_fit[i],n)
            ###############################################################            
            if temp_fit > ls_fit[i]:
                ls_pop[i] = copy.deepcopy(temp_x)
                ls_fit[i] = copy.deepcopy(temp_fit)
#    # 3.5 局部搜索
#    ls_pop = copy.deepcopy(Para_pop)
#    ls_fit = copy.deepcopy(Para_fit)
#    ls_fit_sort=sorted(ls_fit)
#    ls_fit_sort_10=ls_fit_sort[90:]
#    for fit_i in ls_fit_sort_10:
#        i=ls_fit.index(fit_i)
#        ###############################################################
##       [temp_x,temp_fit] = NWLS(ls_pop[i],ls_fit[i],n,weights,edge_all)
#        ###############################################################            
##       [temp_x,temp_fit] = local_optimization_new(ls_pop[i],ls_fit[i],n)
#        [temp_x,temp_fit]=EMNLS(Gi,ls_pop[i],ls_fit[i],n,higher_weights,edge_all,motif_GG)
#        ###############################################################            
#        if temp_fit > ls_fit[i]:
#            ls_pop[i] = copy.deepcopy(temp_x)
#            ls_fit[i] = copy.deepcopy(temp_fit)    
    # 3.6 updata pop and fit
    pop=copy.deepcopy(ls_pop)
    fit=copy.deepcopy(ls_fit)     
 
    
    # 3.7 记录当代结果  
#        ###############################################
#        real=[]
#        fo = open("Karate_groundtruth.txt",'r')
#        for line in fo:
#            real.append(int(float(line)))
#        fo.close()
#        ################################################
#        #
#        nmi=[]
#        for i in range(NP):
#            temp_nmi = ig.compare_communities(real,pop[i],method='nmi',remove_none=False)
#            nmi.append(temp_nmi)    
#        # 当代种群最优nmi记录
#        best_nmi = max(nmi)  
#        #print('best_nmi =', best_nmi)  
#        best_nmi_history.append(best_nmi)
#        # nmi最大值对应best_nmi_x
#        best_nmi_x = pop[nmi.index(best_nmi)]
#        best_nmi_x_history.append(best_nmi_x) 

    # 当代种群最优Qw
    best_fit = max(fit) 
#    print('best_Qw=', best_fit) 
    best_fit_history.append(best_fit)   
    # 当代种群最优Qw对应个体
    best_fit_x = pop[fit.index(best_fit)]
    best_fit_x_history.append(best_fit_x)
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
    # jishu = exetime-50
    if (exetime>=Gen):
        p=1
    # elif exetime >= 50 and best_fit_history[jishu] == best_fit_history[-1]:#30代不增长默认收敛
    #     print("最后迭代次数为",exetime-49)
    #     p = 1
    else:
       exetime+=1  

df_news = pd.read_table(r'community80.txt', header=None)
real_membership = []
for i in df_news[1]:
    real_membership.append(i)
real_membership = np.array(real_membership)

print('Qw',best_fit)
print("NMI",ig.compare_communities(real_membership,best_fit_x,method='nmi',remove_none=False))

# st3 = process_time()
# print('初始化时间',st2 - st1)
# print('主循环时间',st3 - st2)
# print('复杂度时间',st3 - st2)
# 最终结果输出
#print('exetime =',exetime) 
# print('best_Qw+1 =',best_fit) 
# print('best_Qw =',ig.GraphBase.modularity(Gi,best_fit_x,higher_weights)) 
#    print("NMI",ig.compare_communities(real,best_fit_x,method='nmi',remove_none=False))

# 将membership转化为comm_list
# def membership_to_commlist(bestx):
#     comm_IDs = copy.deepcopy(pd.value_counts(bestx).index)
#     #num_comm = len(comm_IDs)
#     comm_list =[]
#     for i in comm_IDs:
#         nodes = copy.deepcopy(np.where(bestx==i))
#         nodes = copy.deepcopy(nodes[0].tolist())
#         comm_list.append(nodes)
#     return comm_list

#ig.GraphBase.modularity(Gi,best_fit_x ,weights)
#comm_list = membership_to_commlist(best_fit_x)

## 保存实验过程中出现的最优结果
#def text_save(content,filename,mode='a'):
#    # Try to save a list variable in txt file.
#    file = open(filename,mode)
#    for i in range(len(content)):
#        file.write(str(content[i])+'\n')
#    file.close()  
#    
#text_save(weights,'Karate_weights.txt')

#画图显示收敛过程
#plt.figure(figsize=(15,7),dpi=80)
#mainColor = (42/256, 87/256, 145/256, 1) # R,G,B,透明度
#plt.xlabel('Number of iterations',color=mainColor)
#plt.ylabel('Best_fit',color=mainColor)             
#plt.tick_params(axis='x',colors=mainColor)  # 坐标轴颜色
#plt.tick_params(axis='y',colors=mainColor)                           
#data_x = np.arange(1,exetime+1)    
#plt.plot(
#        data_x,
#        best_fit_history,
#        '.',
#        color=mainColor,
#        lineWidth=5
#        )                                                
#plt.grid(True) #设置背景栅格
#plt.show() # 显示图形  
#plt.savefig('motif-MSOSCDbrain.png',dpi=300,bbox_inches = 'tight')

