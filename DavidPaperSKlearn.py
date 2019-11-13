## Sklearn version of the GPR model David was using 

import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process import kernels
import time
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import scipy.io as sio


def sqexpkern(x1, x2, k1, k2 ):
    kern = k2*np.exp(-(np.power((x1 - x2), 2)/(2*k1)) )
    return kern

def gradfunc(xs, X, k1, k2, alpha):
    
    kern = np.zeros((np.shape(X)[0], 1))

    for i in range(np.shape(X)[0]):
    
        kern[i] = k1*np.exp(  -(np.linalg.norm(xs - X[i]))**2/(2*k2)  ) 


    grad = np.matmul((X - xs).T, kern )

    grad = np.matmul(grad, alpha.T)

    grad = grad.sum(axis = 1).reshape(X_train.shape[1],1)
    
    return grad


def alphacalc(X1, X2, y, sigma):
    alkern = np.zeros((np.shape(X1)[0], np.shape(X2)[0]))

    for i in range(np.shape(X1)[0]):
    
        for j in range(np.shape(X2)[0]):
        
            alkern[j][i] = k1*np.exp(  -(np.linalg.norm(X1[i] - X2[j]))**2/(2*k2)  )
    
    alpha = alkern + sigma*np.identity(np.shape(X1)[0])
    
    alpha = np.matmul(  np.linalg.inv(alpha), y   )
    
    return alpha



mat_contents = sio.loadmat('PythonFile.mat')

SNR = mat_contents['SNR']

PdBm = mat_contents['PdBm']

X = PdBm

y = SNR

#X = np.random.rand(10000,10)

#y = np.random.rand(10000,1)

# number of data points in total, train and test 

m = len(X) 

m_train = int(m*0.75)

m_test = int(m*0.25)

# split data into train and test 

X_train = X[0:m_train]

X_test = X[m_train:m]

y_train = y[0:m_train]

y_test = y[m_train:m]


X_small = X[0:100]

y_small = y[0:100]

ymean = np.mean(y)  # check the mean of y -> close ish to 0 so should be okay to leave defaults for GPR 


start = time.time()

#kernel = ConstantKernel(1.0, constant_value_bounds="fixed")*RBF(1.0, length_scale_bounds="fixed")

#gpr = GaussianProcessRegressor( alpha=0, normalize).fit(X, y);

kernel = 1.0**2 * RBF(length_scale=1.0) 

print("Initial kernel: %s" % kernel)

gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.044,
                              normalize_y=False)



gpr.fit(X_train, y_train)

end = time.time()

print(end - start)

prediction = gpr.predict(X_test)

acc = (1 - np.absolute(prediction - y_test))*100

acc = np.mean(acc)

# gradient ascent 

# stuff I need to get: Kernel parameters, alpha

opt_kernel = gpr.kernel_

print("Optimised kernel: %s" % gpr.kernel_)

print("Optimised kernel theta parameters: %s" % gpr.kernel_.theta) 

# Note that the first hyperparameter (the one outside the exponential) is already sq rooted 

hyperparameter = np.exp(gpr.kernel_.theta)

k1 = hyperparameter[0] # v^2

k2 = hyperparameter[1] # l^2 

GPalpha = gpr.alpha_

# try and calculate alpha from scratch 

start = time.time()

alphakern = alphacalc(X_small, X_small , 1)

end = time.time()

print(end-start)

#alkern[i] = k1*np.exp(  -(np.linalg.norm(xs - X[i]))**2/(2*k2)  ) 




# implement grad calculation 

xstar = X_train[0:1] 


grad = gradfunc(xstar, X_train, k1, k2, GPalpha)
#kern = np.zeros((np.shape(X_train)[0], 1))
#for i in range(m_train):
    
#    kern[i] = k1*np.exp(  -(np.linalg.norm(xstar - X_train[i]))**2/(2*k2)  ) 

#grad = np.matmul((X_train - xstar).T, kern )
#grad = np.matmul(grad, GPalpha.T)
#grad = grad.sum(axis = 1).reshape(10,1)

# implement grad ascent 

xstar_opt =  X[0:1] 

num_iters = 1000 

gamma = 1e-7 # learning rate

xstar_rec = np.zeros((num_iters, xstar.shape[1]))

start = time.time()

for i in range(num_iters):
    
    xstar_rec[i] = xstar_opt 
    
    xstar_opt = xstar_opt + gamma*gradfunc(xstar, X_train, k1, k2, GPalpha).T 
    
    
    
    
plt.plot(xstar_rec, 'x')
plt.show()

end = time.time()

print(end - start)









