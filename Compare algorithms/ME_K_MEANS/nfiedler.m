function [x, lambda] = nfiedler(A, tol)
% NFIEDLER returns the fiedler vector of the normalized laplacian of A.

if nargin < 2
    tol = 1e-12;
end

L = nlaplacian(A);
n = size(A, 1);%返回矩阵的行数
[V, lambdas] = eigs(L + speye(n), 2 , 'sa', struct('tol', tol));%V是特征向量 lambdas是特征值（两个）
[~, eig_order] = sort(diag(lambdas));
ind = eig_order(end);%找出第二小的特征值对应的索引
x = V(:, ind);%找出第二小特征值对应的特征向量
x = x ./ sqrt(sum(A, 2));
lambda = lambdas(ind, ind) - 1;
