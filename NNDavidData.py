# script using TF to learn the mapping from X to y in David's dataset

# will try to use gradient descent to optimise the powers, as in J. Zhou et.al. “Robust,compact, and flexible neural model for a fiber Raman amplifier"

# NOTE: requires that tensorflow is installed in your virtual env

########################## imports ######################


import numpy as np 
#import time
import matplotlib.pyplot as plt
import scipy.io as sio
#import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib

# %% ######################### get data ######################

mat_contents = sio.loadmat('PythonFile.mat')

y = mat_contents['SNR'] # measured SNR at the end of the link 
 
X = mat_contents['PdBm'] # 1x10 row vectors of amplifier power values in a 10 span link 

# %% ############### split data into 75% train and 25% test ###############

m = len(X) 
m_train = int(m*0.75)
m_test = int(m*0.25)

X_train = X[0:m_train]
X_test = X[m_train:m]
y_train = y[0:m_train]
y_test = y[m_train:m]


# %% ################ build the model using Keras #################

num_hidden_units = 32

input_size = X.shape[1]

# as first layer in a sequential model:
model = Sequential()
dense_1 = Dense(num_hidden_units, input_shape=(input_size,), activation='sigmoid' )
model.add(dense_1)
# now the model will take as input arrays of shape (*, 10)
# and output arrays of shape (*, 32), where * is the batch size
# this allows the batch size to be varied 

# after the first layer, you don't need to specify
# the size of the input anymore:
dense_2 = Dense(1, activation='linear')
model.add(dense_2)

# compile the model 
model.compile(optimizer='rmsprop',
              loss='mse')

# %% ############# fit the model and return predictions ##############
batch_size = 32
model.fit(X, y, epochs=50, batch_size=batch_size, verbose=1)
y_pred = model.predict(X_test, verbose=0)
accuracy = np.mean(abs(y_pred - y_test))
accper = 100 - (accuracy/np.mean(y))*100 # estimate of the accuracy of the model (rough)

# %% ############## implement gradient descent to find the optimised powers ####################

# using method from J. Zhou et.al. “Robust,compact, and flexible neural model for a fiber Raman amplifier"

# extract the weights and biases of the layers 

W1 = dense_1.get_weights()[0]
B1 = dense_1.get_weights()[1].reshape(num_hidden_units,1)
W2 = dense_2.get_weights()[0]
B2 = dense_2.get_weights()[1].reshape(1,1)

# %% derivatives of activation functions 

def ReLUder(x, size):
    # note that changing the value of the der at x = 0 changes the behaviour significantly 
    # should try sigmoid instead 
    x = x*np.identity(size)
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigder(x, size):
    der = (1 - (1/(1 + np.exp(-x) )))*(1/(1 + np.exp(-x) ))
    der = der*np.identity(size)
    return der

# size here is equal to the size (no. of units) of the hidden layer 
    
def gradfunc(x, size ):
    # transform from the power vector to W*x + bias
    x = np.matmul(W1.T, x.T) + B1
    #grad = np.matmul(W2.T, ReLUder(x, size)) 
    grad = np.matmul(W2.T, sigder(x, size)) 
    grad = np.matmul(grad, W1.T)
    return grad
#  ############## implement gradient ascent ##############
def gradascent(x_ini, gamma, num_iters, size  ):
    x = x_ini
    x_rec = np.zeros((num_iters, x_ini.shape[1]))
    for i in range(num_iters):
        x_rec[i] = x
        x = x + gamma*gradfunc(x, size)
    return x, x_rec

gamma = 1
num_iters = 1000
mat_contentsxstar = sio.loadmat('optimised_xstar.mat')
x_opt_GP = mat_contentsxstar['xstar']
x_ini = X[0:1]
#x_ini = x_opt_GP
x_opt, x_rec = gradascent(x_ini, gamma, num_iters, num_hidden_units  )
y_final = model.predict(x_opt, verbose=0)
y_final_GP = model.predict(x_opt_GP, verbose=0)

# %% ############ plot learning curve and final power vector plot #############
font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)


plt.plot(x_rec, 'x')
plt.xlabel("No. iters")
plt.ylabel("$P_{ch}$ (dBm)")
plt.show()


x_ind = np.array([1,2,3,4,5,6,7,8,9,10], dtype = int).reshape(1,10)


plt.plot(x_ind, x_opt, 'o')
plt.xlabel("EDFA")
plt.ylabel("$P_{ch}$ (dBm)")
plt.show()








