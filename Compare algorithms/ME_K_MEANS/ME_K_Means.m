% Generate result from Figure 2 in
% Higher-order organization of complex networks. 
% Austin R. Benson, David F. Gleich, and Jure Leskovec.
% Science, 353.6295 (2016): 163--166.

adj = xlsread('result/jazz_M6_adj.xlsx'); 
A = adj;
W =xlsread('result/jazz_M6_motif_adj.xlsx');
n = size(W,1);  % �ڵ���
W1=sparse(W);%ת��Ϊϡ�����
X = KmeansCluster(W1, 9);%��ʵ��������
%disp(size(X))
dlmwrite('result/jazz_M6_coms.txt',X,'precision', '%.0f');