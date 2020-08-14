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

def gradient(X , Y , epoch , step_sizem , reg_cons):
     dimension = X.shape[1]
     c = []
     beta = np.random.random(dimension)
     beta_Ridge = beta
     beta_Lasso = beta
     for i in range(0 , epoch):
         y_hat = X @ beta
         y_hat_Lasso = X @ beta_Lasso
         y_hat_Ridge = X @ beta_Ridge         
         
         error = (Y - y_hat)
         error_Lasso = (Y - y_hat_Lasso)
         error_Ridge = (Y - y_hat_Ridge)
         
         cost_function = (np.dot(error , error))
         cost_function_Lasso = (np.dot(error_Lasso , error_Lasso))+reg_cons*beta_Lasso[1:]
         cost_function_Ridge = (np.dot(error_Ridge , error_Ridge)) + reg_cons*(np.dot(beta_Ridge[1:] , beta_Ridge[1:]))
         
         derivative = -(2*(error @ X))
         derivative_Lasso = -(2*(error @ X)) + reg_cons
         derivative_Ridge = -(2*(error @ X)) + 2*reg_cons*np.sum(beta[1:]) 
         
         beta = beta - step_size*derivative
         beta_Lasso = beta_Lasso - step_size*derivative_Lasso
         beta_Ridge = beta_Ridge - step_size*derivative_Ridge

     return beta ,beta_Lasso,beta_Ridge, cost_function, cost_function_Lasso ,cost_function_Ridge

cosine=[]
cosine_Lasso=[]
cosine_Ridge=[]

data_dimension=[]

noise=[]

dimension = int(input("Enter how many type of variables"))
epoch = int(input("Enter no. of iteractions"))
step_size = float(input("Enter Step Size"))
times = int(input("Enter How many Times you want to run the code"))
reg_cons= float(input('Regularization Constant'))
choice = input('Enter 1 to change data size \n 2 to change Noise')
if choice == '1':
     sigma = float(input('Enter Noise in o/p variable'))
     for i in range (0 , times):
         n = int(input("Enter how many rows of data"))
        
         X , Y , beta_initial = generate(dimension , n , sigma )
         beta_final ,beta_Lasso,beta_Ridge, cost_function, cost_function_Lasso ,cost_function_Ridge = gradient(X , Y , epoch , step_size , reg_cons)
         
         Y_hat = X @ beta_final
         Y_hat_Lasso = X @ beta_Lasso
         Y_hat_Ridge = X @ beta_Ridge
        
         compare_Y = pd.DataFrame({'Y_actual': Y , 'Y_predicted': Y_hat })
         compare_Beta= pd.DataFrame({'Actual_beta':beta_initial, 'beta_final':beta_final , 'beta_Lasso':beta_Lasso,'beta_Ridge': beta_Ridge })
        
         from sklearn.metrics import mean_absolute_error
         print('mean_absolute_error: ',mean_absolute_error(Y , Y_hat))
         print('mean_absolute_error Lasso Regressor: ',mean_absolute_error(Y , Y_hat_Lasso))
         print('mean_absolute_error Ridge Regressor: ',mean_absolute_error(Y , Y_hat_Ridge))
        
         cos = ((beta_initial @ beta_final).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_final @ beta_final).sum())**(1/2))
         cos_Lasso = ((beta_initial @ beta_Lasso).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_Lasso @ beta_Lasso).sum())**(1/2))
         cos_Ridge = ((beta_initial @ beta_Ridge).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_Ridge @ beta_Ridge).sum())**(1/2))

         cosine.append(float(cos))
         cosine_Lasso.append(float(cos_Lasso))
         cosine_Ridge.append(float(cos_Ridge))

         data_dimension.append(n)
     
     plt.plot(data_dimension , cosine ,label ='Linear Regression')
     plt.plot(data_dimension , cosine_Lasso ,label ='Lasso Regression')
     plt.plot(data_dimension , cosine_Ridge , label ='Ridge Regression')
     plt.legend()
     plt.show()
if choice == '2':
     n = int(input("Enter how many rows of data"))
     for i in range (0 , times):
         sigma = float(input('Enter Noise in o/p variable'))
         X , Y , beta_initial = generate(dimension , n , sigma )
        
         beta_final ,beta_Lasso,beta_Ridge, cost_function, cost_function_Lasso ,cost_function_Ridge = gradient(X , Y , epoch , step_size , reg_cons)
        
         Y_hat = X @ beta_final
         Y_hat_Lasso = X @ beta_Lasso
         Y_hat_Ridge = X @ beta_Ridge
        
         compare_Y = pd.DataFrame({'Y_actual': Y , 'Y_predicted': Y_hat })
         compare_Beta= pd.DataFrame({'Actual_beta':beta_initial, 'beta_final':beta_final, 'beta_Lasso':beta_Lasso,'beta_Ridge': beta_Ridge })
         from sklearn.metrics import mean_absolute_error
         print('mean_absolute_error: ',mean_absolute_error(Y , Y_hat))
         print('mean_absolute_error Lasso Regressor: ',mean_absolute_error(Y , Y_hat_Lasso))
         print('mean_absolute_error Ridge Regressor: ',mean_absolute_error(Y , Y_hat_Ridge))
        
         cos = ((beta_initial @ beta_final).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_final @ beta_final).sum())**(1/2))
         cos_Lasso = ((beta_initial @ beta_Lasso).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_Lasso @ beta_Lasso).sum())**(1/2))
         cos_Ridge = ((beta_initial @ beta_Ridge).sum() )/(((beta_initial @ beta_initial).sum())**(1/2)*((beta_Ridge @ beta_Ridge).sum())**(1/2))

         cosine.append(float(cos))
         cosine_Lasso.append(float(cos_Lasso))
         cosine_Ridge.append(float(cos_Ridge))
         
         
         noise.append(sigma)
    
     plt.plot(noise , cosine , label ='Linear Regression')
     plt.plot(noise , cosine_Lasso , label ='Lasso Regression')
     plt.plot(noise , cosine_Ridge , label ='Ridge Regression')
     plt.legend()
     plt.show()

print(len(data_dimension))
