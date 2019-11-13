% function that can record multiple runs of GPR and take the average

function [xstarave, xstars] = multisampleGPR(X, y, num_samples, sample_size, gamma, num_iters)

    xstars = zeros(num_samples,  size(X,2) );
    for i = 1:num_samples
        
        
        rand_ind = randperm(sample_size);

        X_rand = X(rand_ind,:);
        
        y_rand = y(rand_ind,1);
        
        gprMdl = fitrgp(X_rand,y_rand,'KernelFunction', 'squaredexponential'); 
        
        % kernel parameter 1 = l^2, kernel parameter 2 = v^2 
        k1 = gprMdl.KernelInformation.KernelParameters(1); 
        k2 = gprMdl.KernelInformation.KernelParameters(2); 

        alpha = gprMdl.Alpha;
        
        xstartest = X(1,:); 
        
        [xstars(i,:), ~] =  gradascent(X_rand, xstartest, gamma, k1, k2, alpha, num_iters );
        
    end
    
    xstarave = mean(xstars); 
    
    
end