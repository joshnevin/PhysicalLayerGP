function [xstar, xstarrec] =  gradascent(X, xstar_ini, gamma, k1, k2, alpha, num_iters   )
            
    %grad = gradfunc(xstartest, X, k1, k2, alpha ); 
    % initialise gradient ascent
    xstar = xstar_ini; 
    %xstar_ini = X(1,:); 
    %num_iters = 200; 
    xstarrec = zeros(num_iters,  size(xstar,2) ); % variable for recording values of Xstar
    for i = 1:num_iters
        xstarrec(i,:) = xstar; 
        xstar = xstar + gamma*gradfunc(xstar, X, k1, k2, alpha )';
    end


end