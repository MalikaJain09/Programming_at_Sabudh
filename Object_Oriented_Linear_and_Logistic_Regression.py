import numpy as np
import random

class LinearRegression:
    def __init__(self , dimension , independent_variables ,sigma=0.34 ,epoch=10  , step_size=0.0001):
        self.dimension = dimension
        self.n = independent_variables
        self.sigma = sigma
        self.epoch = epoch
        self.step_size = step_size
        self.X , self.Y , self.beta_original = self.generate()
        self.beta_predict , self.cost_function = self.gradient()
        
    
    def generate(self):

         X = np.random.random((self.n, self.dimension+1))
         for i in range(0 , self.n):
             X[i][0] = 1
         e = random.gauss(0 , self.sigma)
         beta = np.random.random(self.dimension+1)
         x_beta= X @ beta
         Y = x_beta+ e
         return X,Y,beta
    
    def gradient(self):
        dimension = self.X.shape[1]
        c = []
        beta = np.random.random(dimension)
        for i in range(0 , self.epoch):
            y_hat = self.X @ beta
            error = (self.Y - y_hat)
            cost_function = (np.dot(error , error))
           
            cost_function.shape
            c.append(cost_function)
            derivative = -(2*(error @ self.X))
            beta = beta - self.step_size*derivative
           
        return beta , cost_function
    
    def accuracy(self , y_hat):
        error = 1/self.dimension*np.sum((self.Y - y_hat)**2)
        return error
    
    def predict(self):
        Y_hat = self.X @ self.beta_predict
        return Y_hat
    
    def cosine_similarity(self):
        cos = ((self.beta_original @ self.beta_predict).sum() )/(((self.beta_original @ self.beta_original).sum())**(1/2)*((self.beta_predict @ self.beta_predict).sum())**(1/2))
        return cos

l = LinearRegression(300 , 7)
y_hat = l.predict()
print('Accuracy: ',l.accuracy(y_hat))
print('CosineSimilarity',l.cosine_similarity())


class LogisticRegression:
    def __init__(self , dimension , independent_variables ,threshold = 0.000001 ,theta=10 ,epoch=10  , step_size=0.0001):
        self.dimension = dimension
        self.n = independent_variables
        self.theta = theta
        self.epoch = epoch
        self.step_size = step_size
        self.threshold = threshold
        self.X , self.Y , self.beta_original = self.generate()
        self.beta_predict , self.cost_function = self.gradient()
    
    def generate(self):
        X = np.random.randn(self.n, self.dimension+1)
        
        for i in range(0 , self.n):
            X[i][0] = 1  
        beta = np.random.random(self.dimension+1)  
        z = X @ beta
        sigmoid = 1/(1 + np.exp(-z))
        Y = np.where(sigmoid >= 0.5 , 1  , 0)    
        c = (n * self.theta)/100
        i = 0
        with np.nditer(Y, op_flags=['readwrite']) as it: 
             for element in it: 
                if element == 1:
                    element[...] = 0
                    i+=1
                if i == c:
                    break
        return X , Y , beta
    
    def gradient(self):
        m = self.X.shape[1]
        n = self.X.shape[0]
        beta = np.random.random(m)
        p = float('inf')
        
        for i in range(0 , self.epoch):
            
            z_hat = np.dot(self.X , beta)
            sigmoid = 1/(1 + np.exp(-z_hat))
               
            new_loss_function = -np.sum(self.Y*np.log(sigmoid) + (1-self.Y)*np.log((1-sigmoid)))/n 
            if abs(p - new_loss_function) < self.threshold:
                print('Total Iterations', i)
                break
            
            p = new_loss_function
            
            derivative = -1/n * np.dot(self.X.T ,self.Y - sigmoid)
            beta = beta - self.step_size*derivative
       
        return beta , new_loss_function
        
    def accuracy(self , Y_hat ):
        
        n= self.Y.shape[0]
        error= np.where(self.Y!=Y_hat, 0, 1)
        return np.count_nonzero(error==1)*100/n
    
    def predict(self):
        b1= np.dot(self.X,self.beta_predict)
        prob= 1/(1 + np.exp(-b1))
        Y_hat= np.where(prob>=0.5,  1,  0)
        return Y_hat
    
    def cosine_similarity(self):
        cos = ((self.beta_original @ self.beta_predict).sum() )/(((self.beta_original @ self.beta_original).sum())**(1/2)*((self.beta_predict @ self.beta_predict).sum())**(1/2))
        return cos

l = LogisticRegression(300 , 7)
y_hat = l.predict()
print('Accuracy: ',l.accuracy(y_hat))
print('CosineSimilarity',l.cosine_similarity())
