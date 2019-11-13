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

%% Use MATLAB library for GP - start with default settings
% this section of code predicts the SNR from an input of the power in each
% span to ~ 96% accuracy 
tic

[gprMdl, aveaccuracy, ypred] = GPR(X,y, X_test, y_test);

toc

%% gradient descent bit 

% kernel parameter 1 = l^2, kernel parameter 2 = v^2 
k1 = gprMdl.KernelInformation.KernelParameters(1); 
k2 = gprMdl.KernelInformation.KernelParameters(2); 

alpha = gprMdl.Alpha;

xstartest = X(1,:); 

gamma = 5e-5; % learning rate 

num_iters = 400;

[xstar, xstarrec] =  gradascent(X, xstartest, gamma, k1, k2, alpha, num_iters ); 

figure
hold on 
plot(xstarrec, 'x')
legend
hold off 


figure
hold on 
plot(xstar, 'o')
hold off 


ystarfinal = predict(gprMdl, xstar);

figure
hold on 
hist(y_test - ypred, 100);
hold off 





