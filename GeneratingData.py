import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_Data(n , m , theta):
    X = np.random.randn(n, m+1)
    
    for i in range(0 , n):
        X[i][0] = 1  
    beta = np.random.random(m+1)  
    z = X @ beta
    sigmoid = 1/(1 + np.exp(-z))
    Y = np.where(sigmoid >= 0.5 , 1  , 0)    
    c = int((n * theta)/100)
    for i in range(0 , c): 
       if Y[i] == 1:
           Y[i]=0
       else:
           Y[i] = 1
    return X , Y , beta

if __name__ == '__main__':

    m = int(input('Enter no. of Independent variable'))
    n = int(input('Enter the datasize'))
    theta = int(input('Enter theta'))
    X , Y , beta = generate_Data(n , m , theta)
