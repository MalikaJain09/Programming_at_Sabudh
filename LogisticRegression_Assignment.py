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
    c = int(n*theta)
    print(c , theta)
    for i in range(0 , c): 
       if Y[i] == 1:
           Y[i]=0
       else:
           Y[i] = 1
    return X , Y , beta

def optimization(X , Y , k , threshold , l_rate):
    m = X.shape[1]
    n = X.shape[0]
    beta = np.random.random(m)
    p = float('inf')
    
    for i in range(0 , k):
        
        z_hat = np.dot(X , beta)
        sigmoid = 1/(1 + np.exp(-z_hat))
           
        new_loss_function = -np.sum(Y*np.log(sigmoid) + (1-Y)*np.log((1-sigmoid)))/n
        
        if abs(p - new_loss_function) < threshold:
            print('Total Iterations', i)
            break
        
        p = new_loss_function
        
        derivative = 1/n * np.dot(X.T ,sigmoid-Y)
        beta = beta - l_rate*derivative
        
    return new_loss_function , beta

cosine=[]
data_dimension=[]
theta_variation=[]

m = int(input('Enter no. of Independent variable'))
c= int(input('Enter Choice 1 to Change No. of Rows \n 2 to Change Theta'))

if c==1:
    theta= float(input('The probability of flipping the label, Y'))
    k = int(input('Enter The No. of interactions'))
    threshold = float(input('Enter Threshold value'))
    l_rate = float(input('Learning Rate')) 

    for i in range(100 , 10000 , 100 ):
        n = i
        X , Y , beta = generate_Data(n , m , theta)
        
        Cost , beta_final = optimization(X , Y , k , threshold , l_rate)
        
        beta_compare = pd.DataFrame({'beta': beta , 'beta_optimised': beta_final})
        print(beta_compare)
        print(Cost)
        dot_beta = np.dot(beta , beta_final)
        given_norm = np.linalg.norm(beta)
        predicted_norm = np.linalg.norm(beta_final)
        cos = dot_beta / (given_norm * predicted_norm)
        
        cosine.append(float(cos))
        data_dimension.append(n)
        
    plt.scatter(data_dimension , cosine )
    plt.show()

if c == 2:
    n = int(input('Enter the datasize'))
    k = int(input('Enter The No. of interactions'))
    threshold = float(input('Enter Threshold value'))
    l_rate = float(input('Learning Rate'))
    theta = 0.001
    while theta <= 0.6:
        
        X , Y , beta = generate_Data(n , m , theta)
        
        Cost , beta_final = optimization(X , Y , k , threshold , l_rate)
        beta_compare = pd.DataFrame({'beta': beta , 'beta_optimised': beta_final})
        
        # dot_beta = np.dot(beta , beta_final)
        # given_norm = np.linalg.norm(beta)
        # predicted_norm = np.linalg.norm(beta_final)
        # cos = dot_beta / (given_norm * predicted_norm)
        cos = ((beta @ beta_final).sum() )/(((beta @ beta).sum())**(1/2)*((beta_final @ beta_final).sum())**(1/2))

        cosine.append(float(cos))
        theta_variation.append(theta)
        theta +=0.01
        
    plt.scatter(theta_variation , cosine )
    plt.show()
    plt.plot(theta_variation , cosine )
    plt.show()

