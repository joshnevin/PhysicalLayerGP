function [gprMdl, aveaccuracy, ypred]  = GPR(X,y,X_test,y_test)

gprMdl = fitrgp(X,y,'KernelFunction', 'squaredexponential'); 

ypred = predict(gprMdl,X_test);

accuracy = abs(y_test - ypred); 

aveaccuracy = (1 - mean(accuracy))*100; 
        
end