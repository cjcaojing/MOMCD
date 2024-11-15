from sklearn import metrics
from sklearn.cluster import KMeans
import scipy
from matrix import *
import copy
import numpy as np
import matplotlib.pyplot as plt
from opfunu.cec_based.cec2008 import F12008
import igraph as ig
import matplotlib.pyplot as plt
import random
from numpy import random
import networkx as nx
import copy
import pandas as pd
from igraph.clustering import VertexClustering
import find_motifs as fm
def cal_Q(G,k):  # 计算Q
    result = []
    for j in range(len(G.nodes)):
        result.append([])
    index = 0
    for node in G.nodes:
        result[index].append(G.nodes[node]['embed'])
        index = index + 1
    result = np.array(result).reshape(len(G.nodes), maxd)

    model = KMeans(n_clusters=k)  # 分为k类
    model.fit(result)
    label = model.labels_
    #转换格式
    Gnodes = list(G.nodes())
    tmp = dict()
    for i in range(len(label)):
        if label[i] not in tmp.keys():
            com = []
            com.append(Gnodes[i])
            tmp[label[i]] = com
        else:tmp[label[i]].append(Gnodes[i])
    partition = list(tmp.values())
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            t += len([x for x in G.neighbors(node)])  # G.neighbors(node)找node节点的邻接节点
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q
def motif_xuanze(motif):
    M_ins = nx.Graph()####模体按照Benson论文中设置
    if motif == "motif1":

        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2, 3), (1, 3)])  # 连通
   
    return  M_ins
def motif_jiaquan(motif,total_motif_lists_1):

    '''
    #模体在网络中的数量
    motif_number = fm.total_motif_num(G, motif_1, directed=True, weighted=False)#总模体数
    print('总摸体数量', motif_number)
    '''
    
    # motif = motif_7
    #模体在网络中的节点集合和边集合
    motif_node_set, motif_edge_set = total_motif_lists_1 # 获得所有模体的节点和边
    # print('模体点集合', motif_node_set)
    # print('模体边集合',motif_edge_set)
    motif_node_set = [[int(x) for x in sublist] for sublist in motif_node_set]
    motif_edge_set = [[(int(x), int(y)) for x, y in sublist] for sublist in motif_edge_set]
    edge_all = Gi.get_edgelist()
    df = pd.DataFrame({ 'edge': edge_all})
    # index_edge = list(range(M))
    # a = df[df.edge==(7,8)].index.tolist()[0]
    # print(a)
    #网络中每条边的模体加权
    original_weights=[0] * edge_num
    for i in motif_edge_set:
        for j in i:
            ind = df[df.edge==j].index.tolist()[0]
            original_weights[ind] += 1
    weights = original_weights 
    # weights=[]
    # for each_weight in original_weights:
    #     each_weight += 0
    #     weights.append(each_weight) 
    return weights

def prepare(G,filename):
    file_read = open(filename, "r")
    real_label1 = dict()
    real_label = []
    i=0
    for line in file_read:
        line = line.split('\t')
        real_label1[int(line[0])] = int(line[len(line) - 1].split("\n")[0])
        # real_label1[i] = int(line)
        # i=i+1
        
    test_data_1 = sorted(real_label1.items(), key=lambda x: x[0])
    for i in range(len(test_data_1)):
        real_label.append(test_data_1[i][1])
    real_label = np.array(real_label,dtype=int)

    k = max(real_label)+1 # 社团数量
    # k=3
    print("社团数量："+str(k))

    result = []
    for node in G.nodes:
        result.append(G.nodes[node]['embed'])
    result = np.array(result).reshape(len(G.nodes), maxd)

    print("开始聚类")
    model = KMeans(n_clusters=k)  # 分为k类
    model.fit(result)
    label = model.labels_
    return real_label,label,k

def Feature_Propagation(G,m,threshold):
    # 开始特征传播
    m=scipy.sparse.csr_matrix(m)
    epochs = 100  # 迭代次数
    flag = False
    time_begin = time.time()
    nodes = np.array(G.nodes())
    res = list()

    for node in G.nodes:
        res.append(G.nodes[node]['embed'])

    for epoch in range(epochs):
        thres = []
        for node in G.nodes:  # node为w的下标
            # 暂定用异步更新方式
            vector = copy.copy(res[node])
            for nei in G[node]:
                index1 = np.where(nodes == node)[0][0]
                index2 = np.where(nodes == nei)[0][0]
                
                vector += get_acceptance(m, index1, index2) * G.nodes[nei]['embed']  # 2 cora 0.4369287658923185

            vector = normalizaion_arctan(vector)

            thres.append(sum(abs(vector - res[node])) / 256)
            res[node] = copy.copy(vector)
            G.nodes[node]['embed'] = copy.copy(res[node])
            th = sum(thres) / len(thres)
            # th = np.sqrt(np.sum(np.square(vector-res[node])))

            if th < threshold:
                flag = True
        if flag:
            print("迭代了%d" % (epoch + 1) + "次，结束更新")
            break
        print("第" + str(epoch + 1) + "次迭代结束:"+str(th))
    del epoch, node

if __name__ == '__main__':
    net='_08'
    print(net)
    for i in range(1):
        time_begin = time.time()
        m, G ,t= get_newMatrix("./1000-0.02/network"+net+".txt")
        print("特征矩阵计算完成")
    
        label_file = "./1000-0.02/community"+net+".txt"
        Feature_Propagation(G,m,0.001)
    
        real_label,label,k = prepare(G,label_file)
        # print(label)
        with open("./1000-0.02/label"+net+".txt", 'w') as file:
            for label in label:
                file.write(str(label) + '\n')
        time_end= time.time()
        runtime=time_end-time_begin
        G = nx.read_edgelist("./1000-0.02/network"+net+".txt")
        G = G.to_undirected()  # 转换成无向图
        n = G.number_of_nodes()  # 获取一个Graph对象中node的数量
        # 基于这些连边使用igraph创建一个新网络
        Gi = ig.Graph.Read_Edgelist("./1000-0.02/network"+net+".txt")
        Gi = Gi.subgraph(map(int, G.nodes()))  # wG.nodes()获取一个Graph对象中的node数组,
        Gi = Gi.as_undirected()
        
        edge_num = Gi.ecount()
        motif_ins = motif_xuanze("motif1")
       
        total_motif_nums = fm.total_motif_num(G, motif_ins, weighted=False)
        motif_node_set, motif_edge_set = fm.total_motif_list(G, motif_ins, weighted=False)
    
        total_motif_lists = fm.total_motif_list(G, motif_ins, weighted=False)
    
        weight = motif_jiaquan(motif_ins,total_motif_lists)
        print(len(np.array(label)))
        print(len(weight))
    
        qw = ig.GraphBase.modularity(Gi,label,weights =weight)
    
        nmi = ig.compare_communities( real_label,label , method='nmi', remove_none=False)
        t_list=[]
        t_list.append(runtime)
        qw_list=[]
        qw_list.append(qw)
        nmi_list=[]
        nmi_list.append(nmi)
        
        print(f"{np.mean(qw):.3f}"+'('+f"{np.var(qw):.2e}"+')')

        print(f"{np.mean(nmi):.3f}"+'('+f"{np.var(nmi):.2e}"+')')


        # print(f"{np.mean(runtime):.3f}")
