function [gprMdl, aveaccuracy, ypred]  = GPR(X,y,X_test,y_test)

% need to specify exact, or else matlab will only use up to 2000 examples

gprMdl = fitrgp(X,y,'KernelFunction', 'squaredexponential', 'BasisFunction', 'none', 'FitMethod', 'exact');  

ypred = predict(gprMdl,X_test);

accuracy = abs(y_test - ypred); 

aveaccuracy = (1 - mean(accuracy))*100; 
        
end