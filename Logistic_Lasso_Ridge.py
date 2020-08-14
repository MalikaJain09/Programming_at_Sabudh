import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def randomVar(n, m, theta):
    x1= np.random.randn(n, m)
    unit = np.ones([n,1], dtype = int)
    x= np.append(unit, x1, axis=1)

    # Bias
    b= np.random.random(m+1)
    b1= np.dot(x,b)
    prob= sigmoid(b1)
    y= np.where(prob>=0.5,  1,  0)
    
    return x, y, b

def accuracy(b, x, y):
    b1= np.dot(x,b)
    prob= sigmoid(b1)
    pre_y= np.where(prob>=0.5,  1,  0)
    n= y.shape[0]
    error= np.where(y!=pre_y, 0, 1)
    return np.count_nonzero(error==1)*100/n

def accuracy_unseen_Data(b , n, m):
    x1= np.random.randn(n, m)
    unit = np.ones([n,1], dtype = int)
    x= np.append(unit, x1, axis=1)
    beta= np.random.random(m+1)
    
    z= np.dot(x,beta)
    prob= sigmoid(z)
    y= np.where(prob>=0.5,  1,  0)
    
    per_accuracy = accuracy(b , x , y)
    return per_accuracy

def Cost(x, y, lasso,tuningPar=0, epochs=100, th=0.001, lr=0.01):
    p=float('inf')
    n=x.shape[0]
    m=x.shape[1]
    b=np.random.random(m)
    
    penalty1=penalty = 0
    for i in range(epochs):
        
        if lasso==False:
            penalty=(tuningPar * np.sum(beta[1:n] * beta[1:n]))/(2*n)
            penalty1=(tuningPar*np.sum(b[1:n]))/n
        
        elif lasso==True:
            penalty=(tuningPar * np.sum(beta[1:n]))/n
            penalty1=tuningPar/n

        b1=np.dot(x,b)
        predictedProb= sigmoid(b1)
        cost= -(np.sum(y*np.log(predictedProb)+(1-y)*np.log(1-predictedProb)))/n+penalty
        if abs(p-cost)<=th:
            return cost, b
        p = cost
        cfunc = -1/n * np.dot(x.T,predictedProb-y)+penalty1
        b-=lr*cfunc
    return cost, b

def logisticRegression(x, y): 
    cost , b = Cost(x, y, -1)
    print("Logistic Regression Cost=", cost)
    print("Logistic Regression Beta=", b)
    accuracy(b , x , y)
    return b , accuracy(b , x , y)

def lassoReg(x, y):
    turingPara =[0.000001,0.001, 0.01, 0.05, 0.1 , 0.2 ,0.3 , 0.5 , 0.6 , 0.7 , 0.9  , 1.2  , 10 , 30]
    beta_list =[]
    acc1 =[]
    acc2=[]
    for i in range (0 , len(turingPara)):
        cost , b = Cost(x, y,turingPara[i], True)
        print("Lasso Regression Cost=", cost)
        print("Lasso Regression Beta=", b)
        print()
        beta_list.append(b)
        
    return beta_list

def ridgeReg(x, y):
    turingPara =[0.000001,0.001, 0.01, 0.05, 0.1 , 0.2 ,0.3 , 0.5 , 0.6 , 0.7 , 0.9  , 1.2  , 10 , 30]
    beta_list =[]
    acc1 =[]
    acc2 =[]
    for i in range (0 , len(turingPara)):
        cost , b = Cost(x, y,turingPara[i], False)
        print("Ridge Regression Cost=", cost)
        print("Ridge Regression Beta=", b)
        print()
        beta_list.append(b)
        
    return beta_list

data=randomVar(1000, 5, 10)

n=data[1].shape[0]
trainLen=int(0.4*data[0].shape[0])

beta_original , accu = logisticRegression(x=data[0][0:trainLen], y=data[1][0:trainLen])
print("Logistic Regression Accuracy",accu)

turingPara =[0.000001,0.001, 0.005, 0.01, 0.1 , 0.2 ,0.3 , 0.5 , 0.6 , 0.7 , 0.9  , 1.2  , 10 , 30]
beta_predicted_lasso = lassoReg(x=data[0][0:trainLen], y=data[1][0:trainLen])

temp = [list(beta_original)]
column = ['beta_original']
for i in range (0, len(beta_predicted_lasso)):
    temp.append(list(beta_predicted_lasso[i]))
    column.append(str(turingPara[i]))
data_pd=[]
for i in range(0 , len(temp[0])):
    d=[]
    for j in range (0 , len(temp)):
        d.append(temp[j][i])
    data_pd.append(d)
        
    
comparison_Lasso = pd.DataFrame(data_pd , columns = column)

print("Effect on beta by changing Regularization Constant in Lasso Regression")
print(comparison_Lasso)


beta_predicted_ridge = ridgeReg(x=data[0][0:trainLen], y=data[1][0:trainLen])

temp = [list(beta_original)]
column = ['beta_original']
for i in range (0, len(beta_predicted_ridge)):
    temp.append(list(beta_predicted_ridge[i]))
    column.append(str(turingPara[i]))
data_pd=[]
for i in range(0 , len(temp[0])):
    d=[]
    for j in range (0 , len(temp)):
        d.append(temp[j][i])
    data_pd.append(d)
comparison_Ridge = pd.DataFrame(data_pd , columns = column)
   
print("Effect on beta by changing Regularization Constant in Ridge Regression")
print(comparison_Ridge)


n_groups = 6


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.10
opacity = 0.8

rects1 = plt.bar(index, comparison_Lasso['beta_original'], bar_width,
alpha=opacity,
color='g',
label='beta_original')

rects2 = plt.bar(index + bar_width, comparison_Lasso['1e-06'], bar_width,
alpha=opacity,
label='lamda=1e-06')

rects3 = plt.bar(index +2* bar_width, comparison_Lasso['0.5'], bar_width,
alpha=opacity,
label='lamda=0.5')

rects3 = plt.bar(index + 3* bar_width, comparison_Lasso['0.6'], bar_width,
alpha=opacity,
label='lamda=0.6')

rects3 = plt.bar(index + 4* bar_width, comparison_Lasso['0.7'], bar_width,
alpha=opacity,
label='lamda= 0.7')

rects4 = plt.bar(index +5* bar_width, comparison_Lasso['30'], bar_width,
alpha=opacity,
label='30')

plt.xlabel('beta')
plt.ylabel('beta_value')
plt.title('LASSO Regression: Comparison of Beta on the basis of different lamda')
plt.xticks(index + bar_width, ('b0','b1','b2' , 'b3' , 'b4' , 'b5' ))
plt.legend()

plt.tight_layout()
plt.show()

n_groups = 6


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.10
opacity = 0.8

rects1 = plt.bar(index, comparison_Ridge['beta_original'], bar_width,
alpha=opacity,
color='g',
label='beta_original')

rects2 = plt.bar(index + bar_width, comparison_Ridge['1e-06'], bar_width,
alpha=opacity,
label='lamda=1e-06')

rects3 = plt.bar(index +2* bar_width, comparison_Ridge['0.5'], bar_width,
alpha=opacity,
label='lamda=0.5')

rects3 = plt.bar(index + 3* bar_width, comparison_Ridge['0.6'], bar_width,
alpha=opacity,
label='lamda=0.6')

rects3 = plt.bar(index + 4* bar_width, comparison_Ridge['0.7'], bar_width,
alpha=opacity,
label='lamda= 0.7')

rects4 = plt.bar(index +5* bar_width, comparison_Ridge['30'], bar_width,
alpha=opacity,
label='30')

plt.xlabel('beta')
plt.ylabel('beta_value')
plt.title('RIDGE Regression: Comparison of Beta on the basis of different lamda')
plt.xticks(index + bar_width, ('b0','b1','b2' , 'b3' , 'b4' , 'b5' ))
plt.legend()

plt.tight_layout()
plt.show()


