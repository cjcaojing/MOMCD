%W=cell2mat(struct2cell(load('karate2.mat')));
%W=sparse(W)%ת��Ϊϡ�����
%A=cell2mat(struct2cell(load('karate_2.mat')));
%A=sparse(A)%ת��Ϊϡ�����
load 'celegans_data';
[S,Sbar,conductances,W]=MotifSpectralPartitionM6(A);