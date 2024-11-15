% Main function for MWLP 
% 2019-05-31
clc;
% clear all;
uA=xlsread('result/football_M8_adj.xlsx');
% M = MotifAdjacency(sparse(uA),'M8'); %motif adjacency matrix
%%
% 
%  PREFORMATTED
%  TEXT
% 
M=xlsread('result/football_M8_motif_adj.xlsx');
n = size(M,1); % number of nodes
%lambda = 0.5;
Iter = 300;  % iterations 
for run=1:1:10
    para_cluster = cell(10,1);
    cnt = 1;
    for lambda = 0.1:0.1:0.1  % lambda is a parameter
        disp(lambda);
        clusters = zeros(n, Iter);
        for t = 1:Iter
            clu_sequence = MWLP_new(M,uA,lambda);
            clusters(:,t) = clu_sequence(:,1)'; % needs to revise for different datasets
        end  
        para_cluster{cnt} = clusters(:,Iter);
        cnt = cnt+1;
    end 

    % save('result_MWLP\cluster.mat','para_cluster');
    dlmwrite(strcat('result/football_M8_coms_',strcat(num2str(run),'.txt')),para_cluster{1,1},'precision', '%.0f')
    %*******Debug*******%
    % lambda = 0.8;
    % clu = MWLP_new(M,uA,lambda);
end 