# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:05:52 2020

@author: Administrator
"""
import xlwt
#import xlsxwriter
import time
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
import xlsxwriter

#计算模体结构3-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def m1(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        return len(set(u_friends) & set(v_friends))   
#计算模体结构3-2(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/2为模体数量
def m2(u,v,G):
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
#计算模体结构4-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def m3(u,v,G):
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
def m4(u,v,G):
    m4_count = 0 #计数
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
        m4_count += 0
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
        m4_count+=deta
    return m4_count


#计算模体结构4-3(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def m5(u,v,G):
    m5_count = 0 #计数
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
        m5_count+=0
    else:
        #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
        for i in itertools.combinations(u_mor,2):
            min_pq=min(i[0],i[1])
            max_pq=max(i[0],i[1])  
            if (min_pq,max_pq) in edge_all:
                u_list0.append((min_pq,max_pq))            
        deta1 = 0
        for p,q in u_list0:
            deta1 += 1
        m5_count+=deta1
    if len(v_mor) <= 1:
        m5_count+=0
    else:
        for i in itertools.combinations(v_mor,2):
            min_pq=min(i[0],i[1])
            max_pq=max(i[0],i[1])  
            if (min_pq,max_pq) in edge_all:
                v_list0.append((min_pq,max_pq)) 
        deta2 = 0
        for p,q in v_list0:
            deta2 += 1
        m5_count+=deta2 
    return m5_count
    
#计算模体结构4-4(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/5为模体数量
def m6(u,v,G):
    m6_count = 0 #计数
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if (u_friends == []) or (v_friends == []):
        m6_count+=0
    else:
        cn = list(set(u_friends) & set(v_friends))
        if len(cn) <= 1:
            m6_count+=0
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
            m6_count+=deta
    return m6_count

#计算模体结构4-5(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def m7(u,v,G):
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
def m8(u,v,G):
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
        
def moti_num(motif_func_list,M,all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    
    #计算三阶模体 和M4(a,e,f)和S1和SS5
    ij_participate_motif_number_list=[]
    for i,j in all_edge_G:
        ij_participate_motif_number=motif_func_list[M](i,j,GG)        
        ij_participate_motif_number_list.append(ij_participate_motif_number)
    
    #计算4-2,4-3,4-4 S2-S14 SS1-SS4模体 函数中遍历
#    ij_participate_motif_number_list=four_three_morphology(GG,all_edge_G)       
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
    return bestx
#def save(data,path):
#    f = xlwt.Workbook() # 创建工作簿
#    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True) # 创建sheet
#    [h, l] = Adj.shape # h为行数，l为列数
#    for i in range(h):
#        for j in range(l):
#            sheet1.write(i, j, str(data[i, j]))
#    f.save(path)
def save(data,path):
    f = xlsxwriter.Workbook(path) # 创建工作簿
    f.use_zip64()  # 添加这一行启用 ZIP64 扩展
    sheet1 = f.add_worksheet(u'sheet1') # 创建sheet
    [h, l] = data.shape # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(int(i), int(j), int(data[i, j]))
    f.close()    


# =============================================================================
# 参数设置
# =============================================================================
network_name = "powergrid_0" #网络
M = "M1"  #模体
time_begin = time.time()
#####
G = nx.read_edgelist("../realnetwork/"+network_name+".txt")
G = G.to_undirected()
#生成加权矩阵
Gi=ig.Graph.Read_Edgelist(r"../realnetwork/"+network_name+".txt")
Gi=Gi.subgraph(map(int,G.nodes()))          
Gi=Gi.as_undirected()

benchmark=open("../realnetwork/"+network_name+".txt")
benchmark=benchmark.read()
##根据网络边列表生成图
G1 = nx.Graph()
all_edge_G=[]
liebiao=benchmark.split("\n")  #用回车符分裂字符串
for i in range(0,len(liebiao)-1):       
    liebiao1=liebiao[i].split()     #用空格符分裂0和1
    node1=int(liebiao1[0])          #节点转化为整数
    node2=int(liebiao1[1])
    G1.add_edge(node1,node2)
    all_edge_G.append((node1,node2)) 
#edge_all=copy.deepcopy(all_edge_G)
edge_all=list(G1.edges())
all_edge_G=copy.deepcopy(edge_all)
###
motif_func_list = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6, "M7": m7, "M8": m8}
ij_participate_motif_number_list=moti_num(motif_func_list,M,edge_all)
#for i in range(len(ij_participate_motif_number_list)):
#    ij_participate_motif_number_list[i]=ij_participate_motif_number_list[i]+1
    
for i in range(0,len(all_edge_G)):
    all_edge_G[i]=list(all_edge_G[i])
    all_edge_G[i].append(ij_participate_motif_number_list[i])  

G1.add_weighted_edges_from(all_edge_G) 
###
adj= nx.adjacency_matrix(G1, nodelist=range(len(G1.nodes())), weight=None)#邻接矩阵
adj = adj.todense()
motif_adj= nx.adjacency_matrix(G1, nodelist=range(len(G1.nodes())),weight='weight')#加权python3
motif_adj=motif_adj.todense()
save(adj, "../result/"+network_name + '_'+M+'_adj.xlsx')
save(motif_adj, "../result/"+network_name + '_'+M+'_motif_adj.xlsx')
time_end= time.time()
runtime=time_end-time_begin
print(runtime)
