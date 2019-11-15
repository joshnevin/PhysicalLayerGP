%% GaussianProcess.m 
% script to reproduce the results in David's paper using the GP library
% function fitrgp. The program loads David's data, splits the dataset into
% training and testing sets (75:25 split), calls GPR() to train the GP, then calls  
% gradascent to find the optimal powers in each span, xstar, which also
% calls gradfunc() - a function to compute the gradient for grad ascent


% NOTES
% was calculating grad ascent wrongly! -> K = k2*exp(-((x1 - x2).^2/(2*k1.^2)));
% this is not the same as taking the Euclidean distance - have now fixed
% this in gradfunc() 

%% load data and deal with some dodgy points

load('SingleChannelRandom_170919_R1_safe.mat')
N=10;
Mm=16004; % could be larger
PdBm=Results.LP(1:Mm,1:N);
SNR=Results.mleSNR(1:Mm,1);
PdBm(Results.LP(1:Mm,11)<-16,:)=[];
SNR(Results.LP(1:Mm,11)<-16,:)=[];
PdBm(SNR<13,:)=[];
SNR(SNR<13,:)=[];


save('PythonFileSNR', '-v7', 'SNR');

save('PythonFile', '-v7', 'SNR', 'PdBm');

%% split up the data into train and test sets

m = size(SNR,1); 

% number of training examples - 75% of the data 
m_train = size(SNR,1)*0.75;

% number of test examples - 25% of the data 
m_test = size(SNR,1)*0.25;

% randomise the selection of the test and training data to avoid biases
k = randperm(m);

krand = k(1:m_train); 
krand2 = k(m_train+1:m); 

X_train = PdBm(krand,:);
y_train = SNR(krand,1);

X_test = PdBm(krand2,:);
y_test = SNR(krand2,1);

X = PdBm; 
y = SNR; 

X_small = PdBm(125,:);
y_small = SNR(125,1); 


%% Use MATLAB library for GP - start with default settings
% this section of code predicts the SNR from an input of the power in each
% span to ~ 96% accuracy 
tic

%[gprMdl, aveaccuracy, ypred] = GPR(X,y, X_test, y_test);

toc

%% gradient descent bit 

% kernel parameter 1 = l^2, kernel parameter 2 = v^2 
k1 = gprMdl.KernelInformation.KernelParameters(1); 
k2 = gprMdl.KernelInformation.KernelParameters(2); 

alpha = gprMdl.Alpha;

%save('PythonFilealpha', '-v7', 'alpha');

%xstar_ini = xstar_multi_ave; 

xstar_ini = X(1,:);

gamma = 1; % learning rate 

num_iters = 1000;

tic

% can change to David's implmentation by changing the function called by
% gradascent 

[xstar, xstarrec] =  gradascent(X, xstar_ini, gamma, k1, k2, alpha, num_iters ); 

toc 

figure
hold on 
plot(xstarrec, 'x')
title('Learning curve')
legend
hold off 


figure
hold on 
plot(xstar, 'o')
title('Optimal power array')
ylim([0 inf])
hold off 


ystarfinal = predict(gprMdl, xstar);

%{
figure
hold on 
histogram(y_test - ypred, 100);
hold off 
%}

%% Multisample GPR test - 64 samples of size 250 (125 pairs)



% parameters for multisample GPR
num_samples = 64;

sample_size = 250; 

gamma_multi = 1; 

num_iters_multi =  1000; 

%[xstarave, xstars] = multisampleGPR(X, y, num_samples, sample_size, gamma_multi, num_iters); 


% put all of this in a loop -> assign each iter to pre-defined array 


tic

xstar_multi = zeros(num_samples,  size(xstar,2));

for i = 1:num_samples

    rand_ind = randperm(sample_size);

    X_rand = X(rand_ind,:);
        
    y_rand = y(rand_ind,1);
        
    xstar_multi_ini = X_rand(1,:);


    gprMdl_multi = fitrgp(X_rand,y_rand,'KernelFunction', 'squaredexponential','BasisFunction', 'none', 'FitMethod', 'exact'); 


     % kernel parameter 1 = l^2, kernel parameter 2 = v^2 
    
    k1_mul = gprMdl_multi.KernelInformation.KernelParameters(1); % fit based on each sample 
    k2_mul = gprMdl_multi.KernelInformation.KernelParameters(2); 

    alpha_mul = gprMdl_multi.Alpha;
     
    [xstar_multi(i,:), xstar_rec_multi] = gradascent(X_rand, xstar_multi_ini, gamma_multi, k1_mul, k2_mul, alpha_mul, num_iters_multi );

end

toc

% find the average over all of the returned optimal power vectors 

xstar_multi_ave = mean(xstar_multi); 

ystarfinal_multi = predict(gprMdl, xstar_multi_ave);

figure
hold on 
plot(xstar_rec_multi, 'x')
title('Learning curve multi')
hold off 


figure
hold on 
plot(xstar_multi_ave, 'o')
title('Optimal power array multi')
ylim([0 inf])
hold off 



%% gradient test section 

%{

altestgrad = rand(size(X_test,1) , 1); 

gradtestJosh_g = gradfunc(xstar_ini, X_test, 1.5, 2.5, altestgrad);

gradtestDavid_g = gradfunc_david(xstar_ini, X_test, 1.5, 2.5, altestgrad);

ratio = gradtestJosh./gradtestDavid; 



%}

