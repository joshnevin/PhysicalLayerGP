function grad = gradfunc_david(xstar, X, k1, k2, al)

% david's stuff 

K = (k2^2)*exp(-sum((permute(X,[1,3,2])-permute(xstar,[3,1,2])).^2,3)/(2*(k1^2) ));

dk_xdxt = (1/(k1^2))*(X-xstar).*K;

grad = dk_xdxt'*al;

end