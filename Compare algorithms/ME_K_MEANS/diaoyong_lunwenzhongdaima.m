%W=cell2mat(struct2cell(load('karate2.mat')));
%W=sparse(W)%转化为稀疏矩阵
%A=cell2mat(struct2cell(load('karate_2.mat')));
%A=sparse(A)%转化为稀疏矩阵
load 'celegans_data';
[S,Sbar,conductances,W]=MotifSpectralPartitionM6(A);