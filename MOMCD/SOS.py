# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:04:38 2022

@author: WYW
"""
import numpy as np
import pandas as pd 
import random  as rd
import copy
import math
import operator

# 引入外部函数
import find_motifs as fm

# C函数
import cython_function as cfunc

# =============================================================================
#     fit_Qs: 计算种群中每个个体的模糊重叠社区划分的模块度函数Q值
#     Qs: 根据pop计算的模块度值
#     pop: 种群
#     adj: 网络邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#    flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
# =============================================================================
def fit_Qs(Qs,pop,adj,n,c,NP,flag):
    W = np.sum(adj) # 权值之和
    m = np.sum(adj, axis=0) # adj 各列之和
    for N in range(NP):
        #计算每个个体的适应度函数值Q
        U = pop[:,:,N]
        # print(U)
        Q = fit_Q(U,adj,n,c,W,m,flag)
        Qs.append(Q)

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
#     flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
#     return 返回Q值
# =============================================================================
def fit_Q(X,adj,n,c,W,m,flag):
    Q=0
    ###Q###
    mod = np.argmax(X, axis=0).astype('int32')
    Q = cfunc.fit_Q(X,adj,n,c,W,m,mod) # 替换成你需要的QW函数
    return Q

# =============================================================================
#     init_pop: 种群初始化
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#     return: 返回初始化后的种群
# =============================================================================
def init_pop(n,c,NP):
    pop = np.empty((c,n,NP), dtype = float) 
    for N in range(NP):
        for i in range(n):
            membershipList = []
            for k in range(c):
#                rd.seed(N+i+k)#随机种子固定随机数
                membershipList.append(rd.random())
            memberships = np.asarray(membershipList)
            memberships = memberships/sum(memberships)  #归一化
            pop[:,i,N]=memberships
    return pop

def bias_init_pop(pop,n,c,NP,adj):
    bias_pop = copy.deepcopy(pop)
    for N in range(NP):
        # 在该个体中，选择一个节点，将其隶属度赋值给所有相邻节点
        i_node = rd.randint(0, n-1)
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(np.ravel(adj[i_node,:]))
        for j in j_m_nodes[0]:
            bias_pop[:,j,N] = bias_pop[:,i_node,N]
    return bias_pop
       

# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     X: 个体隶属度矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
#     return: 约束检测并修正后的个体
# =============================================================================
def bound_check_revise(X,n,c):
    # 将每个元素约束到[0，1],并归一化
    for i in range(n):
        for k in range(c):
            Xki = X[k,i]
            if Xki > 1:
               Xki = 0.9999
            elif Xki < 0:
                X[k,i] = 0.0001
        X[:,i] = X[:,i] / sum(X[:,i])
    return X

# =============================================================================
# SOSFCD: 共生生物搜索算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: (加权)网络邻接矩阵
# Q_flag: 选择的模块度函数 Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def SOSFCD(pop, fit_values, n, c, NP, adj, Q_flag):
    # Mutualism【互利共生】
    W = np.sum(adj) # 权值之和
    m = np.sum(adj, axis=0) # adj 各列之和
    mutu_pop = copy.deepcopy(pop)
    mutu_fit = copy.deepcopy(fit_values)
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(mutu_fit)
        best_fit_index = mutu_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 互利共生算法
        Xbest = mutu_pop[:,:,best_fit_index]
        Xi = mutu_pop[:,:,i]
        Xj = mutu_pop[:,:,j]
        mutual_vector = 0.5 * (Xi + Xj) # 互利共生向量
        BF1=round(1+rd.random())
        BF2=round(1+rd.random())
        # 生成Xinew和Xjnew
        Xinew = Xi + rd.random()*(Xbest - BF1*mutual_vector)
        Xjnew = Xj + rd.random()*(Xbest - BF2*mutual_vector)
        # 边界约束检查与修正
        Xinew = bound_check_revise(Xinew,n,c)
        Xjnew = bound_check_revise(Xjnew,n,c)
        # 适应度函数值计算
        Xinew_fit = fit_Q(Xinew,adj,n,c,W,m,Q_flag)
        Xjnew_fit = fit_Q(Xjnew,adj,n,c,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if Xinew_fit > mutu_fit[i]:
            mutu_pop[:,:,i] = Xinew    # 保存优秀个体
            mutu_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            # better_number+=1
        if Xjnew_fit > mutu_fit[j]:
            mutu_pop[:,:,j] = Xjnew    # 保存优秀个体
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
        j = rd.choice(ij_list)
        # 共栖算法
        Xbest = comm_pop[:,:,best_fit_index]
        Xi = comm_pop[:,:,i]
        Xj = comm_pop[:,:,j]
        Xinew = Xi + rd.uniform(-1, 1)*(Xbest - Xj)
        # 边界约束检查与修正
        Xinew = bound_check_revise(Xinew,n,c)
        # 适应度函数值计算
        Xinew_fit = fit_Q(Xinew,adj,n,c,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if Xinew_fit > comm_fit[i]:
            comm_pop[:,:,i] = Xinew    # 保存优秀个体
            comm_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            # better_number+=1
    # print("comm_better_number={}".format(better_number))
    # print("comm_best_Q={}".format(max(comm_fit)))
   
    # Parasitism【寄生】
    para_pop = comm_pop
    para_fit = comm_fit
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(para_fit)
        best_fit_index = para_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 寄生算法
        para_vector = copy.deepcopy(para_pop[:,:,i])   # 寄生向量
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] # 随机选择一些节点
        # 在约束范围内随机化节点对应的隶属度值
        para_vector[:,pick] = init_pop(len(pick),c,1)[:,:,0] 
        # 边界约束检查与修正
        para_vector = bound_check_revise(para_vector,n,c)
        # 适应度函数值计算
        para_vector_fit = fit_Q(para_vector,adj,n,c,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if para_vector_fit > para_fit[i]:
            para_pop[:,:,i] = para_vector    # 保存优秀个体
            para_fit[i] = para_vector_fit # 保存优秀个体的适应度函数值
            # better_number+=1
    # print("para_better_number={}".format(better_number))
    # print("para_best_Q={}".format(max(para_fit)))
    # 返回当前进化后的种群和适应的函数值
    return (para_pop, para_fit)


# =============================================================================
#     NMM: 基于邻居节点的社区修正（仅基于边邻居节点）
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体邻接矩阵
#     threshold_value: 阈值
#     Q_flag: 模块度函数选择标识
#     nmm_pop: NMM种群
#     nmm_fit: NMM种群中个体对应的模块度函数值
# =============================================================================
def NMM(pop, fit, n, c, NP, adj, motif_adj, threshold_value, Q_flag):
    nmm_pop = copy.deepcopy(pop)
    nmm_fit = []
    for i in range(NP):
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] #随机选择一定数量的节点
        # pick = seeds #选取全部节点
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        NMM_CD_func(unreasonableNodes, pick, nmm_pop[:,:,i], adj, c, n, threshold_value)
        # 获得该节点应划分的社区号
        node_cno_list=[]
        NMM_P_func(node_cno_list, unreasonableNodes, nmm_pop[:,:,i],adj)
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cno_list,nmm_pop,i)
    # 计算该种群的适应度函数值
    fit_Qs(nmm_fit,nmm_pop,motif_adj,n,c,NP,Q_flag)   #适应度函数值计算
    # 选择优秀个体并保留到种群
    for index in range(NP):
        if nmm_fit[index] > fit[index]:
            pop[:,:,index] = nmm_pop[:,:,index]    #保存优秀个体
            fit[index] = nmm_fit[index] #保存优秀个体的适应度函数值
    return (pop, fit)

# =============================================================================
#     find_unreasonableNodes: 寻找基于边的不合理划分的节点
#     unreasonableNodes: 划分不合理的节点 []
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     threshold_value: 阈值
# =============================================================================
def NMM_CD_func(unreasonableNodes,pick,Xi,adj,c,n,threshold_value):
    for i in pick:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_nodes
        j_nodes = np.nonzero(adj[i,:])[1]
        # 如果 i 节点无邻居节点，则跳过该节点
        if len(j_nodes) == 0:
            continue
        # 获得基于边的邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # i_c != j_c
        cd_i = np.where(j_nodes_c != i_node_c)[0].shape[0] / len(j_nodes)
        # 如果节点 i 划分不合理程度大于阈值，则返回 i
        if cd_i > threshold_value :
            unreasonableNodes.append(i)
            
# =============================================================================
#     NMM_P_func: 寻找节点应划分的社区号
#     node_cno_list: 节点及对应的划分社区[(i,ck)]
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
# =============================================================================
def NMM_P_func(node_cno_list,nodes,Xi,adj):
    for i in nodes:
        # 获得 i 基于边的邻接节点
        j_nodes = np.nonzero(adj[i,:])[1]
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # print("j_nodes_c=",j_nodes_c)
        node_cno_list.append((i,rd.choice(j_nodes_c)))  # choice() 依概率选择
        # i_c = np.argmax(np.bincount(j_nodes_c)) # 直接选择概率最大的社区作为i节点划分的社区
        # node_cno_list.append((i,i_c))

# =============================================================================
#     unreasonableNodes_revise: 修正节点社区编号
#     node_cno_list: 节点和社区编号
#     nmm_pop: nmm种群
#     N: 种群中的第N个个体的序列号
# =============================================================================
def unreasonableNodes_revise(node_cno_list,nmm_pop,N):
    for i_c in node_cno_list:
        i = i_c[0]
        c = i_c[1]
        new_num = nmm_pop[c,i,N] + 0.5
        if new_num > 1.0:
            nmm_pop[c,i,N] = 0.9999
        else:
            nmm_pop[c,i,N] = new_num
        nmm_pop[:,i,N] /= np.sum(nmm_pop[:,i,N]) # 归一化
        
# =============================================================================
#     NWMM_nc_revise: NWMM修正节点社区编号
#     node_cno_list: 节点和社区编号
#     nmm_pop: nmm种群
#     N: 种群中的第N个个体的序列号
# =============================================================================
def NWMM_nc_revise(c,node_cps,nmm_pop,N):
    for i in node_cps.keys():
        cps = node_cps[i]
        for c_p in cps:
            nmm_pop[c_p[0],i,N] = c_p[1]
        # 将i节点的非邻接社区隶属度值调成0
        cs = [cp[0] for cp in cps] # i节点邻接社区
        cset = set([i for i in range(c)])-set(cs)
        for u_c in cset:
            nmm_pop[u_c,i,N] = 0.0
            
 
# =============================================================================
#     choice_by_probability: 依概率选择
#     c_p_list: 节点 i 的候选社区概率列表
#     return : 选择的社区c
# =============================================================================
def choice_by_probability(c_p_list):
    num = 1000
    choice_list = []
    for c_p in c_p_list:
        c = c_p[0]
        p = c_p[1]
        n = int(p*num)
        choice_list += [c]*n
    ic = rd.choice(choice_list) # choice() 依概率选择
    return ic
