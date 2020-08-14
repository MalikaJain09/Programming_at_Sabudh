import numpy as np
import random
from math import isclose
import matplotlib.pyplot as plt
import pandas as pd

def generate(dimension , n , sigma ):

     X = np.random.random((n, dimension+1))
     for i in range(0 , n):
         X[i][0] = 1
     e = random.gauss(0 , sigma)
     beta = np.random.random(dimension+1)
     x_beta=X @ beta
     Y = x_beta+ e
     return X,Y,beta

def gradient(X , Y , epoch , step_size):
     dimension = X.shape[1]
     c = []
     beta = np.random.random(dimension)
     for i in range(0 , epoch):
         y_hat = X @ beta
         error = (Y - y_hat)
         cost_function = (np.dot(error , error))
        
         cost_function.shape
         c.append(cost_function)
         derivative = -(2*(error @ X))
         # print("derivative" , derivative)
         beta = beta - step_size*derivative
    
     return beta , cost_function

cosine=[]
data_dimension=[]
noise=[]
dimension = int(input("Enter how many type of variables"))
epoch = int(input("Enter no. of iteractions"))
step_size = float(input("Enter Step Size"))
times = int(input("Enter How many Times you want to run the code"))
choice = input('Enter 1 to change data size \n 2 to change Noise')
if choice == '1':
     sigma = float(input('Enter Noise in o/p variable'))
     for i in range (0 , times):
         n = int(input("Enter how many rows of data"))
        
         X , Y , beta_initial = generate(dimension , n , sigma )
         beta_final , cost_function = gradient(X , Y , epoch , step_size)
         Y_hat = X @ beta_final
        
         compare_Y = pd.DataFrame({'Y_actual': Y , 'Y_predicted': Y_hat })
         compare_Beta= pd.DataFrame({'Actual_beta':beta_initial, 'beta_final':beta_final })
        
         from sklearn.metrics import mean_absolute_error
         print('mean_absolute_error: ',mean_absolute_error(Y , Y_hat))
        
         cos = ((beta_initial @ beta_final).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_final @ beta_final).sum())**(1/2))
         cosine.append(float(cos))
         data_dimension.append(n)
     plt.plot(data_dimension , cosine )
     plt.show()
if choice == '2':
     n = int(input("Enter how many rows of data"))
     for i in range (0 , times):
         sigma = float(input('Enter Noise in o/p variable'))
         X , Y , beta_initial = generate(dimension , n , sigma )
        
         beta_final , cost_function = gradient(X , Y , epoch , step_size)
        
         Y_hat = X @ beta_final
        
         compare_Y = pd.DataFrame({'Y_actual': Y , 'Y_predicted': Y_hat })
         compare_Beta= pd.DataFrame({'Actual_beta':beta_initial, 'beta_final':beta_final })
         from sklearn.metrics import mean_absolute_error
         print('mean_absolute_error: ',mean_absolute_error(Y , Y_hat))
        
         cos = ((beta_initial @ beta_final).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_final @ beta_final).sum())**(1/2))
    
     cosine.append(float(cos))
     noise.append(sigma)
    
     plt.plot(noise , cosine)
     plt.show()

print(len(data_dimension))
