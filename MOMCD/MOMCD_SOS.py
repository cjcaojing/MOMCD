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
from pandas import DataFrame
from numpy import mean,std,median
import matplotlib.pyplot as plt
from math import exp,log,sqrt 
from time import process_time

# C函数
import cython_function as cfunc

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

   

#EMNCM操作，针对于m2等非全联通模体我们要考虑模体邻居社区，模体邻居节点
def EMNCM_addMotifneighborhood(X,fit,n,NP,edge_all,GG1,motif_matrix,lower_weights):
    #对fit进行排序，选择前50%的个体对应的最小适应度值
    indexs = [k for k in range(NP)]
    # fit_dict = dict(zip(fit,indexs))
    # fit_indes_list = sorted(fit_dict.items(), key=lambda x:x[0],reverse=True)
    #
    # fit_indexs = dict(fit_indes_list).values() #选择fit前50%的个体
    fit_indexs = indexs #全部个体
############变化##########################
    #基于lower_weights生成低阶权重矩阵
    matrix_lh=motif_matrix+lower_matrix
########################################             
    # 对每个种群个体进行操作
    for i in list(fit_indexs):
        # 在个体i中随机选择get_num个节点进行社区调整
        # get_num = np.random.randint(1, n)#np.random.randint(1, n)
        get_num = n
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
            # node_comm = copy.deepcopy(X[i][node])
            node_comm = X[i][node]
            # node邻域节点的社区编号,[]
            # neigh_nodes_comm = copy.deepcopy(X[i][neigh_nodes])
            neigh_nodes_comm = X[i][neigh_nodes]

            ####################    邻居社区的总权重     #############################
            # 邻域社区列表
            # comm_list = copy.deepcopy(pd.value_counts(neigh_nodes_comm).index) #################计算复杂度过高
            # comm_list = copy.deepcopy(comm_list.tolist()) 
            comm_list = list(set(neigh_nodes_comm)) #邻域社区集合             
            # 邻居社区权重
            comm_weight = []
            # 节点到邻居社区的最短路径
            # allneighcomm_Shortest_path=[]
            # 获得每个社区的节点集合
            cno_nodes, cno_comm_nodes = {}, {}
            # 获得每个社区的节点集合
            nodes_all_arr = np.asarray(X[i])
            for cno in comm_list:
                cno_nodes[cno] = np.where(nodes_all_arr==cno)[0].tolist()
                cno_comm_nodes[cno] = []
                    
            for j_index,j in enumerate(neigh_nodes): #当前neighcomm的所有节点
                cno_comm_nodes[neigh_nodes_comm[j_index]].append(j) 
            # 每一个邻居社区的权重
            for k in range(len(comm_list)):
#                print("k",k)
                # 第k个邻居社区的标号
                neighcomm = comm_list[k]
#                print("neighcomm",neighcomm)
                #属于当前neighcomm的所有节点
                # neighcommk_allnodes1 = [j for j,x in enumerate(X[i]) if x==neighcomm]    #计算复杂度过高
                neighcommk_allnodes = cno_nodes[neighcomm]
                #如果节点node在neighcomm内，那么neighcommk_allnodes去除节点node
                if node in neighcommk_allnodes:
                    neighcommk_allnodes.remove(node)
                    
                # 属于当前neighcomm的所有邻居节点
                # neighcomm_nodes_index = [j for j,x in enumerate(neigh_nodes_comm) if x==neighcomm] 
                neighcomm_nodes = cno_comm_nodes[neighcomm]
                # neighcomm_nodes = []               
                # for index in neighcomm_nodes_index:
                #     neighcomm_nodes.append(neigh_nodes[index])    

                # 属于当前neighcomm的每个邻居节点，对应边权重
                neighcomm_nodes_weight = []
                for neighcomm_node in neighcomm_nodes:
####################变化##########################################
                    neighcomm_nodes_weight.append(matrix_lh[node,neighcomm_node])
#############################################################
                # print("neighcomm_nodes_weight",neighcomm_nodes_weight)
                # 当前neighcomm的总权重
                # comm_weight.append(np.sum(neighcomm_nodes_weight))
                comm_weight.append(sum(neighcomm_nodes_weight))

            #每个邻居社区对节点node的吸引力
            # attr_node_neighcomm=[comm_weight[i]/allneighcomm_Shortest_path[i] for i in range(len(allneighcomm_Shortest_path))]    
            #不考虑最短路径
            attr_node_neighcomm = comm_weight
            #节点node对每个邻居社区的隶属度  
            # attr_node_neighcomm_sum=np.sum(attr_node_neighcomm)
            attr_node_neighcomm_sum = sum(attr_node_neighcomm)
            belongdegree_node_neighcomm = [i/attr_node_neighcomm_sum for i in attr_node_neighcomm] 
            # avg_belongdegree_node_neighcomm = sum(belongdegree_node_neighcomm)/len(belongdegree_node_neighcomm)
#            print("attr_node_neighcomm",attr_node_neighcomm)
#            print("belongdegree_node_neighcomm",belongdegree_node_neighcomm)
            ################################################################### 
            #判断节点i是否跟邻居属于一个社区，如果是，那么判断当前社区隶属度是否大于均值，若小于，社区划分不合理
            #判断节点i是否跟邻居属于一个社区,如果，社区划分不合理
            if node_comm not in comm_list or belongdegree_node_neighcomm[comm_list.index(node_comm)]<=0.7:
                # 候选社区集合
                comm_id_for_choice = comm_list
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
            # print("X",X)                   
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
    oldcomm_neighnodes = [i_neighbors[index] for index in oldcomm_neighnodes_index] 
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

# =============================================================================
# 用于计算单个社区节点调整前后的dalte_QWl值
# =============================================================================
def differ_Ql(motif_adj, i, nodes, W, m):
    after_cno_nodes=[]
    if i in nodes:
        after_cno_nodes = copy.deepcopy(nodes)
        after_cno_nodes.remove(i)
    else:
        after_cno_nodes = copy.deepcopy(nodes)
        after_cno_nodes.append(i)  
    befor_cno_arr = np.asarray(nodes, dtype=int)
    befor_cno_Ql = cfunc.fit_Ql(motif_adj, befor_cno_arr, len(nodes), W, m) #移动节点前Ql值
    
    if len(after_cno_nodes)==0: return 0 - befor_cno_Ql
    after_cno_arr = np.asarray(after_cno_nodes, dtype=int)
    after_cno_Ql = cfunc.fit_Ql(motif_adj, after_cno_arr, len(after_cno_nodes), W, m) #移动节点后Ql值
    dalte_cno_Ql = after_cno_Ql - befor_cno_Ql #移动i节点前后icno社区的Ql值的变化
    # print("after_cno_Ql={},befor_cno_Ql={},dalte_cno_Ql={}".format(befor_cno_Ql,after_cno_Ql,dalte_cno_Ql))
    return dalte_cno_Ql

def EMNLS(Gi,bestx,bestfit,n,higher_weights,edge_all,motif_GG,W,m,motif_adj):
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
            # comm_id_i_choice = pd.value_counts(comm_id_i_neighbors).index
            # comm_id_i_choice = comm_id_i_choice.tolist()
            comm_id_i_choice = list(set(comm_id_i_neighbors))
            if comm_id_i in comm_id_i_choice:
                comm_id_i_choice.remove(comm_id_i)  
            # 获得i节点所在社区及其各个邻域社区的节点集
            cno_iset = {}
            cno_iset[comm_id_i] = []
            for cno in comm_id_i_choice: cno_iset[cno] = []
            cno_set = cno_iset.keys()
            for i_node,i_cno in enumerate(bestx): 
                if i_cno in cno_set:
                    cno_iset[i_cno].append(i_node)
            
            #逐个更换节点i的社区标号
            dalte_icno_Ql = differ_Ql(motif_adj, i, cno_iset[comm_id_i], W, m) #i节点移动前后的Ql(icno)
            #迁移到每个邻居社区的增量Q列表
            x_change_diff_fit_list=[]
            for j, jcno in enumerate(comm_id_i_choice):                
                # x_change_diff_fit=differ_Qw(bestx,i,comm_id_i,comm_id_i_choice[j],higher_weights,edge_all,comm_id_i_neighbors,i_neighbors,motif_GG)
                dalte_jcno_Ql = differ_Ql(motif_adj, i, cno_iset[jcno], W, m)
                # print("x_change={},ij_change={}".format(x_change_diff_fit, dalte_icno_Ql + dalte_jcno_Ql))
                x_change_diff_fit_list.append(dalte_icno_Ql + dalte_jcno_Ql)
            # 比较新个体与原个体的优劣
            if max(x_change_diff_fit_list) > 0:
                x_change_diff_fit_list_index = x_change_diff_fit_list.index(max(x_change_diff_fit_list))
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

# =============================================================================
# MOMCD高阶社区检测
# =============================================================================
# 网络信息
network_name = "10"
NMIflag = 1 # 0:关，1：开

G = nx.read_edgelist("LFR1000/network" + network_name +".txt")
G = G.to_undirected()
n=G.number_of_nodes()
dim=n
#获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist("LFR1000/network" + network_name +".txt")
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

############## 网络真实社区划分  ###################
real_membership = []
if NMIflag ==1:
    df_news = pd.read_table("LFR1000/community" + network_name + ".txt", header=None)
    for i in df_news[0]:
        real_membership.append(i)
###########################################################
#EMNCM操作：基于加权矩阵转化加权网络,用于求节点间最短路径
#模体邻接矩阵(考虑了基于模体的邻居节点,用于社区修正操作)
motif_matrix=moti_num_motifadd(edge_all)
motif_adj = motif_matrix.astype(np.int)
W = np.sum(motif_adj) # 权值之和
m = np.sum(motif_adj, axis=0) # adj 各列之和
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

#EMNLS操作：基于higher_weights列表生成加权网络
motif_GG=nx.Graph()
motif_GG.add_nodes_from(range(0,n))
for i in range(len(edge_all)):
    motif_GG.add_weighted_edges_from([(edge_all[i][0],edge_all[i][1],higher_weights[i])])
############################################################

init_start_time = process_time()
# 标准SOS算法参数设置
NP = 10  # The number of candidate solutions
Gen = 2  # The number of iterations
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
Qws,nmis = [],[]
for run in range(10):
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
    init_end_time = process_time()
    # =============================================================================
    # 时间统计列表初始化
    # =============================================================================
    init_spendtime = init_end_time - init_start_time #本次运行程序初始化消耗时间
    GSI_SOS_spendtimes = [] #每次迭代算法种群进化部分消耗时间列表
    ENNCM_spendtimes = []  #每次迭代算法ENMCM操作消耗时间列表
    NCLS_spendtimes = []   #每次迭代算法NCLS操作消耗时间列表
    fit_spendtimes = []   #每次迭代算法目标函数计算消耗时间列表
    MOMCD_spendtimes = []   #每次算法迭代总消耗时间列表
    ##
    print("#### MOMCD_{}_start ####".format(run+1))
    while p<1:
        # # 输出当前进化代数exetime
        # print ('exetime=', exetime)
        # =============================================================================
        #     GSI-SOS种群优化
        # =============================================================================
        MOMCD_start_time = process_time()
        SOS_start_time = process_time()
        # SOS中，目标函数耗时通统计
        SOS_fit_spendtimes = [] #本次迭代中SOS
        
        # 3.1 Mutualism 
        # Mutu_pop=copy.deepcopy(pop)
        # Mutu_fit=copy.deepcopy(fit)
        # for i in range(NP):
           # print(i)
           # (1)Xbest 
    #        best_fit_Mutu = max(Mutu_fit)
    #        best_x_Mutu = Mutu_pop[Mutu_fit.index(max(Mutu_fit))]
    #        # (2)Xj~=Xi
    #        j = np.random.permutation(np.delete(np.arange(NP), i))[0]
    #        # (3)Mutual Vector & Beneficial Factor
    #        mutual_vector = sum(Mutu_pop)/NP
    #        BF1 = BFMIN+(BFMAX-BFMIN)*(2-exp(exetime*log(2)/Gen))
    #        BF2 = BFMIN+(BFMAX-BFMIN)*(exp(exetime*log(2)/Gen)-1)
    #        #BF1, BF2 = np.random.randint(1, 3, 2)
    #        # (4)Xinew
    #        Xinew_Mutu = Mutu_pop[j] + BF1 * (best_x_Mutu - Mutu_pop[i]) + BF2 * (mutual_vector - Mutu_pop[i])
    #        # 限定取值范围
    #        Xinew_Mutu = Xinew_Mutu.round().astype(int) # 取整，变为离散整数
    #        # 每一维元素限定至[0,n)之间
    #        Xinew_Mutu = bound_SOSCD(l_bound, u_bound, Xinew_Mutu, n)
    #        # evaluate Xinew
    #        #Xinew_fit_Mutu = ig.GraphBase.modularity(Gi,Xinew_Mutu)
    #        fit_s1_start = process_time()
    #        Xinew_fit_Mutu = ig.GraphBase.modularity(Gi,Xinew_Mutu,higher_weights)
    #        fit_s1_end = process_time()
    #        SOS_fit_spendtimes.append(fit_s1_end-fit_s1_start)
    #        # (5)updata Mutu_pop and Mutu_fit
    #        if Xinew_fit_Mutu > Mutu_fit[i]:
    #            Mutu_pop[i] = Xinew_Mutu
    #            Mutu_fit[i] = Xinew_fit_Mutu       
               
    #     # 3.2 Commensalism 
    #     Comm_pop=copy.deepcopy(Mutu_pop)
    #     Comm_fit=copy.deepcopy(Mutu_fit)
    #     for i in range(NP):
    #        # (1)Xbest 
    #        best_fit_Comm = max(Comm_fit)
    #        bad_fit_Comm = min(Comm_fit)
    #        best_x_Comm = Comm_pop[Comm_fit.index(max(Comm_fit))]
    #        # (2)Xj~=Xi
    #        random_three=np.random.permutation(np.delete(np.arange(NP), i))
    #        j_first = random_three[0]
    #        j_second = random_three[1]
    #        j_third = random_three[2]
    #        Q1=np.percentile(Comm_fit,25)
    #        Q3=np.percentile(Comm_fit,75)
    #        # (3)Xinew 
    #        array_rand = (Comm_fit[i]-bad_fit_Comm)/(max(Comm_fit)-bad_fit_Comm + 0.000001)
    #        if Comm_fit[i]>=Q3:           
    #            Xinew_Comm = Comm_pop[i] + array_rand * (Comm_pop[j_second] - Comm_pop[j_third])     
    #        elif Comm_fit[i]<=Q1:
    #            Xinew_Comm = Comm_pop[j_first] + array_rand * (best_x_Comm - Comm_pop[i])
    #        else:
    #            Xinew_Comm = Comm_pop[i] + array_rand * (best_x_Comm - Comm_pop[j_first])
    #        # 限定取值范围
    #        Xinew_Comm = Xinew_Comm.round().astype(int) # 取整，变为离散整数
    #        # 每一维元素限定至[0,n)之间
    #        Xinew_Comm = bound_SOSCD(l_bound, u_bound, Xinew_Comm, n)
    #        # evaluate Xinew 
    #        #Xinew_fit_Comm = ig.GraphBase.modularity(Gi,Xinew_Comm)
    #        fit_s2_start = process_time()
    #        Xinew_fit_Comm = ig.GraphBase.modularity(Gi,Xinew_Comm,higher_weights)
    #        fit_s2_end = process_time()
    #        SOS_fit_spendtimes.append(fit_s2_end - fit_s2_start)
    #        # (5)updata Comm_pop and Comm_fit
    #        if Xinew_fit_Comm > Comm_fit[i]:
    #            Comm_pop[i] = Xinew_Comm
    #            Comm_fit[i] = Xinew_fit_Comm
    
    #     # 3.3 Parasitism
    #     Para_pop=copy.deepcopy(Comm_pop)
    #     Para_fit=copy.deepcopy(Comm_fit)
    #     for i in range(NP):
    #        best_fit_Para = max(Para_fit)
    #        best_x_Para_index=Para_fit.index(max(Para_fit))
    #        best_x_Para = Para_pop[best_x_Para_index]
    #        # (1) Xj~=Xi
    #        j = np.random.permutation(np.delete(np.arange(NP), i))[0]
    #        # (2) Parasite Vector 
    #        Para_vector=copy.deepcopy(Para_pop[i])
    #        # 方式1：仅随机更改一维分量
    #        # Para_vector[np.random.randint(0, dim)] = float_rand(l_bound, u_bound,1)
    #        # 方式2：随机更改多维分量 
    #        change_number = np.random.randint(1, n)
    #        pick = np.random.permutation(np.arange(n))[0:change_number]
    #        for index in pick:
    #            Para_vector[index] = np.random.randint(n)      
    #        # 限定取值范围
    #        Para_vector = Para_vector.round().astype(int) # 取整，变为离散整数
    #        # 每一维元素限定至[0,n)之间
    #        Para_vector = bound_SOSCD(l_bound, u_bound, Para_vector, n)       
    #        # (3) evaluate Parasite Vector 
    #        fit_s3_start = process_time()
    #        Para_vector_fit = ig.GraphBase.modularity(Gi,Para_vector,higher_weights)
    #        fit_s3_end = process_time()
    #        SOS_fit_spendtimes.append(fit_s3_end - fit_s3_start)
    #        # (4) evaluate IF
    #        IF=(Para_vector_fit-min(Para_fit))/(max(Para_fit)-min(Para_fit)+0.000001)
    # #       IF=(max(Para_fit)-Para_vector_fit)/(max(Para_fit)-min(Para_fit)+0.000001)
    #        bestnew_x_Para= best_x_Para+IF*(Para_vector-best_x_Para)
    #        # 限定取值范围
    #        bestnew_x_Para = bestnew_x_Para.round().astype(int) # 取整，变为离散整数
    #        # 每一维元素限定至[0,n)之间
    #        bestnew_x_Para = bound_SOSCD(l_bound, u_bound, bestnew_x_Para, n)
    #        fit_s4_start = process_time()
    #        bestnew_x_Para_fit = ig.GraphBase.modularity(Gi,bestnew_x_Para,higher_weights)
    #        fit_s4_end = process_time()
    #        SOS_fit_spendtimes.append(fit_s4_end - fit_s4_start)
    #        if bestnew_x_Para_fit > best_fit_Para:
    #            Para_pop[best_x_Para_index] = bestnew_x_Para
    #            Para_fit[best_x_Para_index] = bestnew_x_Para_fit
        mutu_pop = copy.deepcopy(pop)
        mutu_fit = copy.deepcopy(fit)
        # better_number = 0
        for i in range(NP):
            # 找到当代种群中的最优个体
            best_fit = max(mutu_fit)
            best_fit_index = mutu_fit.index(best_fit) 
            # Xi != Xj
            ij_list = [i for i in range(NP)]
            ij_list.remove(i)
            j = np.random.choice(ij_list)
            # 互利共生算法
            Xbest = mutu_pop[:,:,best_fit_index]
            Xi = mutu_pop[:,:,i]
            Xj = mutu_pop[:,:,j]
            mutual_vector = 0.5 * (Xi + Xj) # 互利共生向量
            BF1=round(1+np.random.random())
            BF2=round(1+np.random.random())
            # 生成Xinew和Xjnew
            Xinew_Mutu = Xi + np.random.random()*(Xbest - BF1*mutual_vector)
            Xjnew_Mutu = Xj + np.random.random()*(Xbest - BF2*mutual_vector)
            # 边界约束检查与修正
            Xinew_Mutu = Xinew_Mutu.round().astype(int) # 取整，变为离散整数
            Xjnew_Mutu = Xjnew_Mutu.round().astype(int)
            # 每一维元素限定至[0,n)之间
            Xinew_Mutu = bound_SOSCD(l_bound, u_bound, Xinew_Mutu, n)
            Xjnew_Mutu = bound_SOSCD(l_bound, u_bound, Xjnew_Mutu, n)
            # 适应度函数值计算
            Xinew_fit = ig.GraphBase.modularity(Gi,Xinew_Mutu,higher_weights)
            Xjnew_fit = ig.GraphBase.modularity(Gi,Xinew_Mutu,higher_weights)
            # 选择优秀个体并保留到种群
            if Xinew_fit > mutu_fit[i]:
                mutu_pop[:,:,i] = Xinew_Mutu    # 保存优秀个体
                mutu_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
                # better_number+=1
            if Xjnew_fit > mutu_fit[j]:
                mutu_pop[:,:,j] = Xjnew_Mutu    # 保存优秀个体
                mutu_fit[j] = Xjnew_fit # 保存优秀   个体的适应度函数值
                # better_number+=1
        # print("mutu_better_number={}".format(better_number))
        # print("mutu_best_Q={}".format(max(mutu_fit)))
        
        # Commensalism【共栖】
        comm_pop = mutu_pop
        comm_fit = mutu_fit
        # better_number = 0
        for i in range(NP):
            # 找到当代种群中的最优个体
            best_fit = max(comm_fit)
            best_fit_index = comm_fit.index(best_fit) 
            # Xi != Xj
            ij_list = [i for i in range(NP)]
            ij_list.remove(i)
            j = np.random.choice(ij_list)
            # 共栖算法
            Xbest = comm_pop[:,:,best_fit_index]
            Xi = comm_pop[:,:,i]
            Xj = comm_pop[:,:,j]
            Xinew_Comm = Xi + np.random.uniform(-1, 1)*(Xbest - Xj)
            # 边界约束检查与修正
            Xinew_Comm = Xinew_Comm.round().astype(int) # 取整，变为离散整数
            Xinew_Comm = bound_SOSCD(l_bound, u_bound, Xinew_Comm, n)
            # 适应度函数值计算
            Xinew_fit = ig.GraphBase.modularity(Gi,Xinew_Comm,higher_weights)
            # 选择优秀个体并保留到种群
            if Xinew_fit > comm_fit[i]:
                comm_pop[:,:,i] = Xinew_Comm    # 保存优秀个体
                comm_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
                # better_number+=1
        # print("comm_better_number={}".format(better_number))
        # print("comm_best_Q={}".format(max(comm_fit)))
       
        # Parasitism【寄生】
        Para_pop = comm_pop
        Para_fit = comm_fit
        # better_number = 0
        for i in range(NP):
            # 找到当代种群中的最优个体
            best_fit = max(Para_fit)
            best_fit_index = Para_fit.index(best_fit) 
            # Xi != Xj
            ij_list = [i for i in range(NP)]
            ij_list.remove(i)
            j = np.random.choice(ij_list)
            # 寄生算法
            para_vector = copy.deepcopy(Para_pop[:,:,i])   # 寄生向量
            seeds = [i for i in range(n)]
            np.random.shuffle(seeds)
            pick = seeds[:np.random.randint(1, n)] # 随机选择一些节点
            # 在约束范围内随机化节点对应的隶属度值
            para_vector[np.random.randint(0, dim)] = float_rand(l_bound, u_bound,1)
            # 边界约束检查与修正

            para_vector = para_vector.round().astype(int) # 取整，变为离散整数
            para_vector = bound_SOSCD(l_bound, u_bound, para_vector, n)
            # 适应度函数值计算
            para_vector_fit = ig.GraphBase.modularity(Gi,para_vector,higher_weights)
            # 选择优秀个体并保留到种群
            if para_vector_fit > Para_fit[i]:
                Para_pop[:,:,i] = para_vector    # 保存优秀个体
                Para_fit[i] = para_vector_fit # 保存优秀个体的适应度函数值




        SOS_end_time = process_time()
        SOS_spend_time = SOS_end_time - SOS_start_time
        # print("SOS_spend_time=",SOS_spend_time)
        
        
        # =============================================================================
        #     EMNCM节点社区修正
        # =============================================================================
        # 3.4 clean_up_NW
        
        EMNCM_start_time = process_time()
        Para_pop_copy = copy.deepcopy(Para_pop)
        Para_fit_copy = copy.deepcopy(Para_fit)
        # 对Para_pop_copy中每个个体进行节点修正
        ###################################################################################################
        # 针对种模块度值在前30%的个体进行EMNCM操作
        Para_clean_pop = EMNCM_addMotifneighborhood(Para_pop_copy,Para_fit_copy,n,NP,edge_all,GG1,motif_matrix,lower_weights)
        EMNCM_end_time = process_time()
        EMNCM_spend_time = EMNCM_end_time - EMNCM_start_time
        # print("EMNCM_spend_time=",EMNCM_spend_time)
        ENNCM_spendtimes.append(EMNCM_spend_time)
        
        
        # =============================================================================
        #     优秀个体选择（目标函数值计算）
        # =============================================================================
        # 计算适应度值
        fit_start_time = process_time()
        Para_clean_fit = []
        for i in range(NP):        
            #############################################################################        
            Para_clean_fit.append(ig.GraphBase.modularity(Gi,Para_clean_pop[i],higher_weights))
            #############################################################################
        fit_end_time = process_time()
        EMNCM_fit_spent_time = fit_end_time - fit_start_time
        # 根据Para_clean_pop，更新 Para_pop 和 Para_fit
        better_number=0
        for i in range(NP):
            # 统计上述操作中，改进个体数目
            if Para_clean_fit[i] > Para_fit[i]:
                better_number = better_number+1
            # 即使Qw值是保持不变的，也保留节点修正结果，使其与邻域节点在一个社区，提高NMI
            # >=
            if Para_clean_fit[i] >= Para_fit[i]:
                Para_pop[i] = Para_clean_pop[i]
                Para_fit[i] = Para_clean_fit[i]  
        # print("####适应度值计算end####")
        
        # =============================================================================
        #     EMNLS局部搜索操作
        # =============================================================================
        # 3.5 局部搜索
        # print("局部搜索开始！")
        LS_start_time = process_time()
        ls_pop = copy.deepcopy(Para_pop)
        ls_fit = copy.deepcopy(Para_fit)
        
        better_number=0
        #针对前10%的个体进行局部搜索
        #对fit进行排序，选择前10%的个体对应的最小适应度值
        indexs = [k for k in range(n)]
        fit_dict = dict(zip(fit,indexs))
        fit_indes_list = sorted(fit_dict.items(), key=lambda x:x[0],reverse=True)
        fit_indexs = dict(fit_indes_list[0:NP//10]).values() #选择fit前10%的个体
        # mean_fit=mean(ls_fit)
        for i in list(fit_indexs):
            ###############################################################
            [temp_x,temp_fit] = EMNLS(Gi,ls_pop[i],ls_fit[i],n,higher_weights,edge_all,motif_GG,W,m,motif_adj)
            ###############################################################            
            # [temp_x,temp_fit] = local_optimization(ls_pop[i],ls_fit[i],n)
            ###############################################################            
            if temp_fit > ls_fit[i]:
                ls_pop[i] = temp_x
                ls_fit[i] = temp_fit
        LS_end_time = process_time()
        LS_spend_time = LS_end_time - LS_start_time
        # print("LS_spend_time=",LS_spend_time)
        NCLS_spendtimes.append(LS_spend_time)
        SOS_fit_spend_time = sum(SOS_fit_spendtimes)
        GSI_SOS_spendtimes.append(SOS_spend_time - SOS_fit_spend_time) #GSI-SOS花费时间（除去目标函数运行时间）
        fit_spendtimes.append(EMNCM_fit_spent_time  + SOS_fit_spend_time) #目标函数本次迭代运行总耗时
        
        # =============================================================================
        #     保留优秀个体到下代种群
        # =============================================================================
        # 3.6 updata pop and fit
        pop=copy.deepcopy(ls_pop)
        fit=copy.deepcopy(ls_fit)     
        MOMCD_end_time = process_time()
        MOMCD_spend_time = MOMCD_end_time - MOMCD_start_time
        MOMCD_spendtimes.append(MOMCD_spend_time)
        # 当代种群最优Qw
        best_fit = max(fit) 
    #    print('best_Qw=', best_fit) 
        best_fit_history.append(best_fit)   
        # 当代种群最优Qw对应个体
        best_fit_x = pop[fit.index(best_fit)]
        best_fit_x_history.append(best_fit_x)
        if exetime % 10 == 0: 
            print('exetime={},ceshibest_Qw={}'.format(exetime,best_fit)) 
        # 当代种群最优个体,对应nmi
        if NMIflag ==1:
            best_fit_x_nmi = ig.compare_communities(real_membership,best_fit_x,method='nmi',remove_none=False)
            best_fit_x_nmi_history.append(best_fit_x_nmi)
            # print('best_fit_x_nmi =',best_fit_x_nmi)
    
                 
        # 3.8 whether the loop stop?  
        #if (exetime>=Gen) or (abs(abs(best_x_with_nmi) - optimal_nmi) <= 1.0e-4):
        # print("####MOMCD end####")
        # 输出数据
        jishu = exetime-30
        if (exetime>=Gen) or (abs(abs(best_fit) - optimal_Qw) <= 1.0e-5):
            p=1
            print('exetime={},ceshibest_Qw={}'.format(exetime,best_fit)) 
            if NMIflag ==1:
                print("NMI=",best_fit_x_nmi)
        elif exetime > 30 and (best_fit_history[jishu] == best_fit or (abs(best_fit_history[jishu] - best_fit) <= 1.0e-7)):#30代不增长默认收敛
            p = 1    
            print("最后迭代次数为",exetime-30)
            print('exetime={},ceshibest_Qw={}'.format(exetime,best_fit)) 
            if NMIflag ==1:
                print("NMI=",best_fit_x_nmi)
        else:
           exetime+=1  
    
    # =============================================================================
    # 算法各部分时间汇总及所得结果统计展示    
    # =============================================================================
    #总耗时（包括后30次迭代）
    # print("\n########### 算法耗时统计（包括后30次迭代） ##############")
    # print('初始化消耗时间：', init_spendtime)
    # print('GSI-SOS消耗时间：', sum(GSI_SOS_spendtimes))
    # print('ENMCM操作消耗时间：', sum(ENNCM_spendtimes))
    # print('NCLS操作消耗时间：', sum(NCLS_spendtimes))
    # print('目标函数计算消耗时间：', sum(fit_spendtimes))
    # print('算法总消耗时间：', sum(MOMCD_spendtimes))
    # # 总耗时（不包括后30次迭代）
    # print("\n########### 算法耗时统计（不包括后30次迭代） ##############")
    # print('初始化消耗时间：', init_spendtime)
    # print('GSI-SOS消耗时间：', sum(GSI_SOS_spendtimes[:-30]))
    # print('ENMCM操作消耗时间：', sum(ENNCM_spendtimes[:-30]))
    # print('NCLS操作消耗时间：', sum(NCLS_spendtimes[:-30]))
    # print('目标函数计算消耗时间：', sum(fit_spendtimes[:-30]))
    # print('算法总消耗时间：', sum(MOMCD_spendtimes[:-30]))
    
    print("########### 算法结果统计##############")
    print('Qw={}'.format(best_fit))
    if NMIflag ==1:
        print('NMI={}'.format(max(best_fit_x_nmi_history)))
        nmis.append(max(best_fit_x_nmi_history))
    Qws.append(best_fit)
# =============================================================================
# 数据输出
# =============================================================================
print("######## {} #######".format(network_name))        
print('Qw_mean={},std={}, max={}'.format(np.mean(Qws), np.std(Qws), max(Qws)))
if NMIflag ==1:
    print('NMI_mean={},std={}, max={}'.format(np.mean(nmis), np.std(nmis), max(nmis)))

