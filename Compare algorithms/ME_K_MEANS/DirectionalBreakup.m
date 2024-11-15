function [B, U, G] = DirectionalBreakup(A)
% DIRECTIONALBREAKUP returns the bidirectional, unidirectional, and
% undirected versions of the adjacency matrix A.
%
% [B, U, G] = DirectionalBreakup(A) returns
%   B: the bidirectional subgraph
%   U: the unidirectional subgraph
%   G: the undirected graph
%
%  Note that G = B + U

A(find(A)) = 1;%矩阵的非零元素都变为1 
B = spones(A&A');  % bidirectional A'表示转置 spones函数判断存在双向的连边
U = A - B; % unidirectional
G = A | A';
