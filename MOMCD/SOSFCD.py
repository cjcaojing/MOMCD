# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    FFM_CD_main_v1_NMM
   使用各种优化算法，基于模体的加权网络的社区检测
"""

import numpy as np
import igraph as ig
import networkx as nx
import os
import time
import copy
from tqdm import tqdm

# 各模块函数
import motif_network_construct as net_stru_func
import SOS as func

# 引入外部函数
import find_motifs as fm

# =============================================================================
# 网络信息 
# network
# =============================================================================
## 真实网络
path = r"./"
## 真实网络
lesmis_network = path + r'football.txt'

# 选择网络
# real
network = lesmis_network
network_name = 'football'
groundtruth_path = path + "/real/" + network_name + '_groundtruth.txt'

# 获取网络数据中的边列表，并根据其使用igraph创建网络
G1 = nx.read_edgelist(network, create_using=nx.Graph())
G1 = G1.to_undirected()
Gi=ig.Graph.Read_Edgelist(network)
Gi=Gi.subgraph(map(int,G1.nodes()))
Gi=Gi.as_undirected()

edge_all = Gi.get_edgelist()
# 各参数设置
# =============================================================================
n=G1.number_of_nodes()
NP = 100
c = 10 #社区的真实划分数
Gen = 50  #进化代数
threshold_value = 0.25 #阈值
# 各标记列表
Mlist = {1:"M1",2:"M2",3:"M3",4:"M4",5:"M5",6:"M6",7:"M7",8:"M8"} #模体选择列表
Qlist = {1:"Q",2:"Qg",3:"Qc_FCD",4:"Qc_OCD",5:"Qov",6:"MC"} # 模块度函数列表
nmmlist = {1:"NOMM",2:"NMM",3:"MNMM",4:"NWMM"} # nmm操作列表
# 本次算法使用的标记
M_flag = Mlist[1]
Q_flag = Qlist[1] #模块度函数 Q
# 独立运行运行次数
Independent_Runs = 1#本次实验独立运行次数
NMIflag = 0 # 0:关闭NMI，1:开启NMI
 
# ================================================== ===========================
# 构建基于模体M1的加权网络
# =============================================================================
G = net_stru_func.construct_weighted_network(Gi,n,M_flag) #构建出基于M_flag模体加权的网络

# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 
    
# 构建基于模体的加权网络邻接矩阵motif_adj
motif_adj = nx.adjacency_matrix(G)
motif_adj = motif_adj.todense() 

# 初始化NMi
nmilist = [] # 用于保存每一代的NMI值
# 获取真实社区划分列表
real_mem = []
if NMIflag==1:
    with open(groundtruth_path, mode='r',encoding='UTF-8') as f:
          real_mem = list(map(int,f.read().splitlines()))

run = 0 # 本程序开始独立运行的次数
while (run < Independent_Runs):
    # 全局变量设置
    pop_best_history = np.zeros((c,n,Gen)) # 用于保存历史最优的个体记录
    best_in_history_Q = [] #用于保存历史最优Q值]
    # =============================================================================
    # 种群初始化，有偏操作
    # =============================================================================
    #种群初始化
    pop = func.init_pop(n, c, NP)  #初始化种群
    fit_values = []
    func.fit_Qs(fit_values,pop,adj,n,c,NP,Q_flag)   #适应度函数值计算

    #有偏操作
    bias_pop = func.bias_init_pop(pop, n, c, NP, adj) #对初始化后的种群进行有偏操作
    bias_fit_values = []
    func.fit_Qs(bias_fit_values,bias_pop,adj,n,c,NP,Q_flag) #适应度函数值计算
    #选择优秀个体并保留到种群
    for index in range(NP):
        if bias_fit_values[index] > fit_values[index]:
            pop[:,:,index] = bias_pop[:,:,index] #保存优秀个体
            fit_values[index] = bias_fit_values[index] #保存优秀个体的适应度函数值
    
    # =============================================================================
    # Main
    #【使用优化算法进行社区检测】
    # =============================================================================
        print("=============net:{0}====C:{1}===================".format(network_name,c))
        start = time.process_time()
        for gen in tqdm(range(Gen)):
            # SOSFCD算法
            (new_pop, new_fit) = func.SOSFCD(pop, fit_values, n, c, NP,adj,Q_flag)
            # NMM操作
            (new_pop, new_fit) = func.NMM(new_pop, new_fit, n, c, NP, adj, adj, threshold_value, Q_flag)
            # 当代最优值
            best_Q = max(new_fit)
            best_index = new_fit.index(best_Q)
            bestx = new_pop[:,:,best_index]
            membership_c = np.argmax(bestx, axis=0)
            # 个体Xbest，并记录最优个体对应的Q值及NMI
            # print("best_Q={}".format(max(fit_values)))
            best_in_history_Q.append(best_Q)
            pop_best_history[:,:,gen] = bestx
            
            # 更新pop,fit
            pop = copy.deepcopy(new_pop)
            fit_values = copy.deepcopy(new_fit)

            if NMIflag==1:
                nmi=ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)    

            if (gen+1) % 10 ==0:
                print("#####"+ M_flag +"_SOSFCD_" + Q_flag + "_#####")
                print("gen=",gen+1)
                print("c_count=",len(set(membership_c)))
                print("membership_c=",list(membership_c))
                if NMIflag==1:
                    print("NMI=",nmi)
                print("best_"+ Q_flag + "=", best_Q)
                
    end = time.process_time()
    print("spend_time=", end - start)