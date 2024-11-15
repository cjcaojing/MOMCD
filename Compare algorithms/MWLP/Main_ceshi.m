% Main function for MWLP 
% 2019-05-31
clc;
clear all;

%load('data\Cornell.mat');
%M = MotifAdjacency(sparse(uA),'M4'); %motif adjacency matrix
%uA = karate();
net=1;
[n,adj]=network_parameters(net);
uA=adj
M = xlsread('karate.xlsx','Sheet1')
n = size(M,1); % number of nodes
%lambda = 0.5;
Iter = 200;
% 10 iterations 
para_cluster = cell(10,1);
cnt = 1;
for lambda = 0.1:0.1:0.1% lambda is a parameter
    disp(lambda);
    clusters = zeros(n, Iter);
    for t = 1:Iter
        clu_sequence = MWLP_new(M,uA,lambda);
        clusters(:,t) = clu_sequence(:,1)'; % needs to revise for different datasets
    end  
    para_cluster{cnt} = clusters;
    cnt = cnt+1;
end

%save('result_MWLP\karate.mat','para_cluster');

%*******Debug*******%
% lambda = 0.8;
% clu = MWLP_new(M,uA,lambda);
