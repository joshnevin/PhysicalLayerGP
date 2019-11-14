function grad = gradfunc(xstar, X, k1, k2, al)

%Ktest = sqexp(xstar,X,k1,k2); 

%Ktest = sum(Ktest,2);

K = zeros(size(X,1),1); 
for i = 1:size(X,1)

    K(i) = k2*exp((-norm(xstar - X(i,:))^2)/(2*k1)); 

end

%grad = (k2/k1)*(X - xstar)'*K*al'; 

grad = (1/k1)*(X - xstar)'*K*al'; 

grad = sum(grad,2);


end