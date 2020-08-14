import numpy as np
import pandas as pd
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
        
        derivative = -1/n * np.dot(X.T ,Y - sigmoid)
        beta = beta - l_rate*derivative
   
    return new_loss_function , beta

if __name__ =='__main__':
    m = int(input('Enter no. of Independent variable'))
    n = int(input('Enter the datasize'))
    theta= float(input('The probability of flipping the label, Y'))
    k = int(input('Enter The No. of interactions'))
    threshold = float(input('Enter Threshold value'))
    l_rate = float(input('Learning Rate'))    
    X , Y , beta = generate_Data(n , m , theta)
    
    Cost , beta_final = optimization(X , Y , k , threshold , l_rate)
    
    beta_compare = pd.DataFrame({'beta': beta , 'beta_optimised': beta_final})
    print(beta_compare)
    print(Cost)
