# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:56:18 2022

@author: WYW
"""
# Cython —— Q函数
import numpy as np
cimport numpy as np 
cimport cython
from libc.math cimport exp,sqrt
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free

# =============================================================================
#     fit_Qg: 计算单个个体的模糊重叠社区划分的模块度函数Qg值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Qg(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double Qg = 0.0
    for k in range(c):
        for i in range(n):
            for j in range(n):
                Qg = Qg + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)*X[k,i]*X[k,j]
    Qg = Qg*1.0/W
    return Qg

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和,即i节点的度
#     mod: membership,社区划分
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Q(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m,long [:] mod):
    cdef double Q = 0.0
    for i in range(n):
        for j in range(n):
            if mod[i] == mod[j]:
                Q = Q + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)
    return Q*1.0/W

# =============================================================================
#     fit_Ql: 计算Ql值
#     adj: 加权邻接矩阵
#     nodes: 社区中的节点列表
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和,即i节点的度
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Ql(long[:,:] adj, long[:] nodes, long n, long W, long[:] m):
    cdef double Ql = 0.0
    for i_index in range(n):
        for j_index in range(n):
            Ql = Ql + (adj[nodes[i_index],nodes[j_index]] - (m[nodes[i_index]]*m[nodes[j_index]])*1.0/W)
    return Ql*1.0/W

# =============================================================================
#     fit_Qc: 计算单个个体的模糊重叠社区划分的模块度函数Qc值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Qc(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double u_ki,u_kj,minu_kij,temp,sij
    cdef double Qc = 0.0
    for i in range(n):
        for j in range(n):
            # 计算sij
            sij= 0.0000
            temp = 0.000
            for k in range(c):
                u_ki = X[k,i]
                u_kj = X[k,j]
                if u_ki<u_kj:
                    temp = sqrt(u_ki)
                else:
                    temp = sqrt(u_kj)
                if temp > sij:
                    sij = temp            
            # 根据sij求Qc
            Qc = Qc + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)*1.0/W*sij
    return Qc


# =============================================================================
#     fit_Qov: 计算单个个体的模糊重叠社区划分的模块度函数Qg值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Qov(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double Qov = 0.0, wSum, lSum
    cdef double** r
    cdef double** w
    cdef int* pointk
    cdef int index, nk
    
    # 创建二维矩阵
    r = <double**>malloc(n * sizeof(double*))
    for i in range(n):
        r[i] = <double*>malloc(n * sizeof(double))
#        memset(r[i], 0, n * sizeof(double))
        
    w = <double**>malloc(n * sizeof(double*))
    for i in range(n):
        w[i] = <double*>malloc(n * sizeof(double))
#        memset(w[i], 0, n * sizeof(double))
        
    for k in range(c):
        #获得第K个社区的所有节点标号(非零元素的下标)
        nk = 0
        for i in range(n):
            if X[k,i] > 0.0:
                nk = nk + 1       
        pointk = <int*>malloc(nk * sizeof(int))
        index = 0
        for i in range(n):
            if X[k,i] > 0.0:
                pointk[index] = i
                index = index + 1 

        # 对矩阵赋值
        for i in range(nk):
            for j in range(nk):
                r[i][j] = 1.0/((1+exp(-(60*X[k,pointk[i]]-30)))*(1+exp(-(60*X[k,pointk[j]]-30))))  
        for i in range(nk):
            for j in range(nk):
                # 求和
                wSum=0
                for t in range(nk):
                    wSum = wSum + r[i][t]
                lSum=0
                for t in range(nk):
                    lSum = lSum + r[t][j]
                # 计算w
                w[i][j] = wSum*lSum*1.0/(nk*nk)
        # 计算Qov值
        for i in range(nk):
            for j in range(nk):
                Qov = Qov + (r[i][j]*adj[pointk[i],pointk[j]] - w[i][j]*(m[0,pointk[i]]*m[0,pointk[j]])*1.0/W)
                
        free(pointk)
    # 释放内存
    for i in range(n):
        free(r[i])
        free(w[i])
    free(r)
    free(w)
    Qov = Qov*1.0/W
    return Qov      

# =============================================================================
#     getMEM_adj: 获得Xi的模体，边及隶属度融合的权重矩阵
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getEdgeW(double[:,:] X, long[:,:] me_adj, long[:,:] node_set, long[:,:,:] edge_set,long i, long j, long c, long lenSet, long M_n, long M_e):
    cdef double m_W = 0.0, tmp_m, sum_membership, sum_w
    cdef int M_c, c_max
    # 创建一个一维数组, 并以0值初始化
    i_M_c = <int*>malloc(c * sizeof(int*))
    for i_c in range(c):
        i_M_c[i_c]=0
    for m_index in range(lenSet): #第m个模体
        # 模体 M1
        # 获得该模体M当前所在的社区
        M_c = 0 #初始化M_c值
        for i_index in range(M_n): #模体中的第i个节点
            membership = 0.0 # 初始化membership
            c_max=0
            for c_index in range(c):#节点i所在的第c个社区
                tmp_m = X[c_index,node_set[i_index,m_index]]
                if tmp_m > membership:
                    membership = tmp_m
                    c_max=c_index         
            i_M_c[c_max] += 1
        for c_index in range(c): #获得模体M所在社区
            if i_M_c[c_index] > M_c:
                M_c = c_index
            i_M_c[c_index] = 0
        # 计算该模体的隶属度之和
        sum_membership=0.0
        for i_index in range(M_n):
            sum_membership += X[M_c,node_set[i_index,m_index]]
        # 计算该模体的权重之和
        sum_w = 0.0 #初始化sum_w值
        for e_index in range(M_e):
            sum_w += me_adj[edge_set[e_index,0,m_index],edge_set[e_index,1,m_index]]
        # 计算该边的融合权重
        m_W += (me_adj[i,j]/sum_w*sum_membership)
    free(i_M_c)
    return me_adj[i,j] + m_W

# =============================================================================
#     renewMUs: 更新MU隶属度矩阵
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
cpdef renewMUs(double[:,:,:] MUs, double[:,:,:] pop, long[:,:] mno_arr, long r, long c, long NP, long D):
    cdef double mmem 
    for N in range(NP):
        for M in range(r):
            for k in range(c):
                mmem = 0.0  #模体M对社区k的隶属度
                for d in range(D):
                    mmem = mmem + pop[k,mno_arr[M,d],N]
                MUs[k,M,N] = mmem/D 
                
# =============================================================================
#     getCDi: 获得节点i的CD(i)值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef getCDI(double[:] result,double[:] attr_cki_arr,long[:,:] mno, long[:] mw, double[:,:] MU, long[:] ms, long m_len, long[:] ijc_nodes_arr, long ijc_len,
                    long[:] j_nodes, long j_len, long[:] j_nodes_c_arr, long[:] j_nodes_c_set, long jsetc_len, long[:,:] motif_adj, 
                    long[:,:] me_adj, long Cnei, long i, long i_node_c, long D):
    cdef double wij_sum = 0.0, wij = 0.0, meij=0.0, meij_sum=0.0
    cdef int jcno, no, mno_jc_w
    # 计算i与其所有邻居节点的权值总和
    for j_index in range(j_len):
        wij_sum = wij_sum + me_adj[i,j_nodes[j_index]] #边权重+模体权重
    # 计算i所在社区的邻居节点权值
    for j_index1 in range(ijc_len):
        wij = wij + me_adj[i,ijc_nodes_arr[j_index1]]
    # print("======wij========",wij)
    if m_len > 0:
        for jsetc_index in range(jsetc_len):
            jcno = j_nodes_c_set[jsetc_index]
            meij = 0.0
            for m_index in range(m_len):
                mno_jc_w = 0 #该模体与jc社区的连接权重
                # print("i={},j_cno={},mno={}".format(i,jcno,mno[ms[m_index]])
                for D_index in range(D):
                    # 查询模体节点是否在jc社区中
                    no = mno[ms[m_index],D_index] #模体节点
                    if no != i: 
                        if j_nodes_c_arr[binarySearch1(j_nodes, j_len, no)] == jcno: #若该节点在社区jc中
                            mno_jc_w = mno_jc_w + motif_adj[i, no]
                meij = meij + MU[jcno,ms[m_index]] * mno_jc_w*1.0/mw[ms[m_index]]
                
            attr_cki_arr[jsetc_index] = meij
            meij_sum = meij_sum + meij
            wij_sum = wij_sum + meij
            if jcno == i_node_c:
                wij = wij + meij
            # print("i={},j_cno={},wij={},wij_sum={},Cnei={}".format(i,j_nodes_c_set[jsetc_index],wij,wij_sum,Cnei))
    # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
    cd_i = wij - wij_sum*1.0/Cnei
    result[0] = cd_i
    result[1] = meij_sum

# =============================================================================
#     getAttrSum: 获得节点i对其各个邻接社区的隶属度以及总隶属度
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getAttrSum(long[:,:] mno,long[:] mw,double[:] arrt_cks_i, long wij_sum, double[:,:] MU,double[:] ic_memberships, long[:] ms, long m_len, long[:] j_nodes, long j_len,
                        long[:] j_nodes_c_set, long[:] j_nodes_c_arr, long jsetc_len, long[:,:] motif_adj, long[:,:] me_adj, long i, long D):
    cdef double arrt_sum = 0.0, arrt_cki=0.0, wij_m=0.0
    # print("==============")
    if m_len > 0:
        for jsetc_index in range(jsetc_len):
            arrt_cki = 0.0
            for m_index in range(m_len):
                wij_m = 0.0 #该模体与jc社区的连接权重
                for D_index in range(D):
                    # 查询模体节点是否在jc社区中
                    if j_nodes_c_arr[binarySearch1(j_nodes, j_len, mno[ms[m_index],D_index])] == j_nodes_c_set[jsetc_index]: #若该节点在社区jc中
                        wij_m = wij_m + motif_adj[i, mno[ms[m_index],D_index]]
                arrt_cki = arrt_cki + MU[j_nodes_c_set[jsetc_index],ms[m_index]] * (wij_m)/mw[ms[m_index]]
            arrt_sum = arrt_sum + arrt_cki
            arrt_cks_i[jsetc_index] = arrt_cki
            # print("i={},j_cno={},wij_sum={},arrt_cki={},arrt_sum={}".format(i,j_nodes_c_set[jsetc_index],wij_sum,arrt_cki,arrt_sum))
    return wij_sum + arrt_sum

# =============================================================================
#     getAttr: 获得节点i的Attr(ck,i)值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getAttrCki(long wij_sum, double mem_sum, double arrt_cki, long[:] ck_nodes, long ckj_len, long[:,:] me_adj, long i):
    cdef double wij_ck = 0.0, arrtCki = arrt_cki
    for ckj_index in range(ckj_len):
        wij_ck = wij_ck + me_adj[i,ck_nodes[ckj_index]]
    return (arrtCki + wij_ck)*1.0/(mem_sum + wij_sum)


# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     X: 个体隶属度矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bound_check_revise(double[:,:] X, long n, long c):
    cdef double Xki_sum
    for i in range(n):
        Xki_sum = 0.0
        for k in range(c):
            if X[k,i] > 1:
                X[k,i] = 0.9999
            elif X[k,i] < 0:
                X[k,i] = 0.0001
            Xki_sum = Xki_sum + X[k,i]
        for ik in range(c):
            X[ik,i] = X[ik,i] / Xki_sum
            
# =============================================================================
#     fai_m: faim值计算
#     faim_flag: 选用方式【1，2】 1:计算整个社区中的最小模体电导，2:计算整个社区划分中的平均模体电导
#     paration: 标签型社区划分
#     parationSet: 由社区划分标签所组成的集合
#     parationSet_len: parationSet集合的长度
#     node_motif_num_list: 节点的点模体数量列表
#     total_motif: 网络中所有的模体集合[[1,2,3],[2,3,4],[2,4,5]]
#     n: 节点数量
#     M_: 模体阶数
#     return faim值
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fai_m(long faim_flag, long[:] partition, long[:] partitionSet, long partitionSet_len, long[:] node_motif_num_list, long[:,:] total_motif, long total_motif_len, long n, long M_):
    cdef int* list1
    cdef int* cno_inum
    cdef int** cno_iset
    cdef double* list2
    cdef double tmp_vlue, list2_minvalue, list2_sum, l1
    cdef int im_num, im_sum_num=0, cno, temp_value, shjian_mt, sqnei_s, sqnei_sfei, flag, fei_flag, i, list2_len
    
    if partitionSet_len != n and partitionSet_len != 1:  #所有节点一个社区 和 每个节点一个社区的情况为1.0
        cno_inum = <int*>malloc(n * sizeof(int*)) #各社区中节点的数目
        for i in range(n):
            cno_inum[i] = 0
            
        for c_index in range(n):
            cno = partition[c_index]
            cno_inum[cno] = cno_inum[cno] + 1
            
        cno_iset = <int**>malloc(n * sizeof(int*)) #各社区对应的节点集合，集合中节点按节点标号大小正序排列
        for cno_index in range(partitionSet_len):
            cno = partitionSet[cno_index]
            cno_iset[cno] = <int*>malloc(cno_inum[cno] * sizeof(int))
            for i_index in range(cno_inum[cno]):
                cno_iset[cno][i_index] = -1 #列表中的值均初始化为-1
                        
        for i in range(n):
            cno = partition[i]
            for i_index in range(cno_inum[cno]):
                if cno_iset[cno][i_index] == -1:
                    cno_iset[cno][i_index] = i    #将节点添加到所属社区中
                    break
        
        list1 = <int*>malloc(n * sizeof(int*)) #单个社区内点模体之和列表
        list2 = <double*>malloc(n * sizeof(double*)) #社区与其余节点  间 的模体
        list2_len = 0 #list2有效大小
        for c_index in range(n):
            list1[c_index]=0
            list2[c_index]=0.0
        
        for cno_index in range(n):  #c_index==i
            cno = partition[cno_index]
            im_num = node_motif_num_list[cno_index]
            list1[cno] = list1[cno] + im_num
            im_sum_num = im_sum_num + im_num

                
        for cno_index in range(partitionSet_len):             
            cno = partitionSet[cno_index]
            temp_value = im_sum_num-list1[cno]
            if temp_value < list1[cno]:
                list1[cno] = temp_value      #最小的单个社区  与  其余节点的点模体作为分母
        im_sum_num=0
        
        for cno_index in range(partitionSet_len):  
            cno = partitionSet[cno_index]
            l1 = list1[cno]
            if l1 != 0 and cno_inum[cno] >= M_:
                sqnei_s = 0
                sqnei_sfei = 0
                for mt_index in range(total_motif_len):
                    flag = 0   #模体中节点在当前社区的数量
                    fei_flag = 0   #模体中节点在非当前社区的数量
                    for i_index in range(M_):
                        i = total_motif[mt_index, i_index]   #获得模体中的节点
                        if binarySearch(cno_iset[cno], cno_inum[cno]-1, i) > -1:
                            flag = flag + 1                   
                        else:
                            fei_flag = fei_flag + 1
                    if flag == M_:
                        sqnei_s = sqnei_s + 1   #若该模体的所有节点均在该社区中，及该模体存在于cno社区中
                    if fei_flag == M_:
                        sqnei_sfei = sqnei_sfei + 1   #若该模体的所有节点均在非该社区中，及该模体存在于非cno社区中
                shjian_mt = total_motif_len - sqnei_s - sqnei_sfei #社区间模体数量
                if shjian_mt != total_motif_len:
                    list2[cno] = shjian_mt/l1
                    list2_len = list2_len + 1
            else:
                list2[cno] = 1.0
                list2_len = list2_len + 1
               
        list2_minvalue = 1.0
        list2_sum = 0.0
        for cno_index in range(partitionSet_len):
            cno = partitionSet[cno_index]
            tmp_vlue = list2[cno]
            if tmp_vlue < list2_minvalue:
                list2_minvalue = tmp_vlue
            list2_sum = list2_sum + tmp_vlue
        # 释放内存
        free(cno_inum)
        free(list1)
        free(list2)
        for i in range(partitionSet_len):
            free(cno_iset[partitionSet[i]])
        free(cno_iset)
        # 返回结果
        if faim_flag == 2:
            return list2_sum / list2_len #fai_m2
        else:
            return list2_minvalue
    else:
        return 1.0

# =============================================================================
#     binarySearch: 二分查找数据,非递归实现
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int binarySearch(int* arr1, int r1, int x1):
    cdef int mid, x=x1, l=0, r=r1
    cdef int* arr=arr1
    while(l<=r):
        mid = int(l + (r - l)/2)
        if x < arr[mid]: 
            r = mid - 1
        elif x > arr[mid]: 
            l = mid + 1
        else: 
            return mid 
    return -1 

# =============================================================================
#     binarySearch: 线性查找数据
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int binarySearch1(long[:] arr, int r1, int x1):
    cdef int index=0,
    for i in range(r1):
        if arr[i] == x1:
            return i
        index = index + 1
    return -1
    

        
        
        

