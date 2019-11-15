function grad = gradfunc(xstar, X, k1, k2, al)

K = zeros(size(X,1),1); 
for i = 1:size(X,1)

    K(i) = (k2^2)*exp((-norm(xstar - X(i,:))^2)/(2*(k1^2)));  
    
end

grad = (1/(k1^2))*(X - xstar)'*(K.*al); 


end