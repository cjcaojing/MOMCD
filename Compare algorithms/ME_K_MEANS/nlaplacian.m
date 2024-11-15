function L = nlaplacian(A)
% NLAPLACIAN returns the normalized laplacian of A

d = sum(A,2)%取每行的和
d = full(d); %稀疏矩阵变为全矩阵
d(d ~= 0) = 1 ./ sqrt(d(d ~= 0));%数除矩阵每个元素用./(点除） sqrt取平方根
[i, j, v] = find(A);
[m,n] = size(A);
L = sparse(i,j,-v.*(d(i).*d(j)), m, n);
L = L + speye(n); % note the negative above
