function [x, lambda] = nfiedler(A, tol)
% NFIEDLER returns the fiedler vector of the normalized laplacian of A.

if nargin < 2
    tol = 1e-12;
end

L = nlaplacian(A);
n = size(A, 1);%���ؾ��������
[V, lambdas] = eigs(L + speye(n), 2 , 'sa', struct('tol', tol));%V���������� lambdas������ֵ��������
[~, eig_order] = sort(diag(lambdas));
ind = eig_order(end);%�ҳ��ڶ�С������ֵ��Ӧ������
x = V(:, ind);%�ҳ��ڶ�С����ֵ��Ӧ����������
x = x ./ sqrt(sum(A, 2));
lambda = lambdas(ind, ind) - 1;
