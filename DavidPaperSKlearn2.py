################ imports ###############

import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
import time
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import scipy.io as sio

################ function defs ###############

# function for calculating the value of alpha from scratch 

def alphacalc(X1, X2, y, sigma):  
    alkern = np.zeros((np.shape(X1)[0], np.shape(X2)[0]))

    for i in range(np.shape(X1)[0]):
    
        for j in range(np.shape(X2)[0]):
        
            alkern[j][i] = k1*np.exp(  -(np.linalg.norm(X1[i] - X2[j]))**2/(2*k2)  )
    
    alpha = alkern + sigma*np.identity(np.shape(X1)[0])
    
    alpha = np.matmul(  np.linalg.inv(alpha), y   )
    
    return alpha


def gradfunc(xs, X, k1, k2, alpha): # verified to be consistent with MATLAB gradfunc
    
    kern = np.zeros((np.shape(X)[0], 1))

    for i in range(np.shape(X)[0]):
    
        kern[i] = (k1**2)*np.exp(  -(np.linalg.norm(xs - X[i]))**2/(2*(k2**2))  ) 


    grad = np.multiply(kern, alpha )


    grad = (1/k2**2)*np.matmul((X - xs).T, grad )

    #grad = np.matmul(grad, alpha.T)

    #grad = (1/k2)*grad.sum(axis = 1).reshape(X_train.shape[1],1)
    
    return grad


def gradascent(X, xstar_ini, gamma, k1, k2, alpha, num_iters  ):
    
    xstar = xstar_ini
    
    xstar_rec = np.zeros((num_iters, xstar.shape[1]))
    
    for i in range(num_iters):
        
        xstar_rec[i] = xstar
        
        xstar = xstar + gamma*gradfunc(xstar, X, k1, k2, alpha).T
    
    return xstar, xstar_rec



################ load the data from v7 .mat file ###############

mat_contents = sio.loadmat('PythonFile.mat')

mat_contentsalpha = sio.loadmat('PythonFilealpha.mat')

mat_contentsxstar = sio.loadmat('optimised_xstar.mat')

xstar_ini = mat_contentsxstar['xstar']

alpha = mat_contentsalpha['alpha']

SNR = mat_contents['SNR']

PdBm = mat_contents['PdBm']

X = PdBm

y = SNR

############### split up the data into train, test and small datasets  ###############

m = len(X) 

m_train = int(m*0.75)

m_test = int(m*0.25)

################ split data into train and test ###############

X_train = X[0:m_train]

X_test = X[m_train:m]

y_train = y[0:m_train]

y_test = y[m_train:m]

X_small = X[0:175]

y_small = y[0:175]
 
############### train the GPR model using the GPR library ###############

# can comment out the train part once the gpr object is in the workspace 

start = time.time()

#kernel = 1.0**2 * RBF(length_scale=1.0) # use RBF kernel as David does in his paper 

kernel = 3.5**2 * RBF(length_scale=26.6) # use starting values close to the optimum 

print("Initial kernel: %s" % kernel)


#gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.044, normalize_y=False)

#gpr.fit(X, y)

print("Optimised kernel: %s" % gpr.kernel_)

end = time.time()

print(end - start)

############### Extract the hyperparameters ############### 

hyperparameter = np.exp(gpr.kernel_.theta)

k1 = hyperparameter[0] # v^2

k2 = hyperparameter[1] # l^2

GPalpha = gpr.alpha_  # values of alpha output by gpr lib 

#GPalphacalculated = alphacalc(X_small, X_small, y_small, 0.044)

############# implement gradient ascent #############

xstar_ini2 = X[0:1]

gamma = 1

num_iters = 50

start = time.time()

[xstar, xstar_rec] = gradascent(X, xstar_ini2, gamma, k1, k2, GPalpha, num_iters  )

end = time.time()

print(end - start)

################ find the predicted optimum SNR ################


ystarfinal = gpr.predict(xstar)

ystarfinalMAT = gpr.predict(xstar_ini)


############# plot learning curve and final power vector plot #############

plt.plot(xstar_rec, 'x')
plt.show()


xstar_ind = np.array([1,2,3,4,5,6,7,8,9,10], dtype = int).reshape(1,10)


plt.plot(xstar_ind, xstar, 'o')
plt.show()













