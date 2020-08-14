import numpy as np
import random
import math
import matplotlib.pyplot as plt

def genrate_data(theta, n, m):
    x = np.random.randn(n , m+1)
    for i in range(0, len(x)):
        x[i][0] = 1
    beta = np.random.random(m+1)
    x_beta =  np.dot(x , beta) #matrix product
    sigmoid = 1 / (1 + np.exp(-x_beta)) #calculate probability i.e. sigmoid function
    y = np.where(sigmoid >=0.5, 1, 0)
    random_number = random.randint(1, 5000)
    # add noise
    for i in range(0, int(theta*n)):
        if y[i] == 0:
            y[i] = 1

        elif y[i] == 1:
            y[i] = 0

    return (x, y, beta)

def logistic_regression(X, Y, epoch, learning_rate, tou):
    n = X.shape[0]
    m = X.shape[1]
    beta = np.random.random(m)
    prev_cost = float('inf')
    for i in range(0, epoch):
        X_beta = np.dot(X , beta)
        sig_value = 1 / (1 + np.exp(-X_beta))
        y = np.where(sig_value > 0.5, 1, 0)
        cost = -(np.sum(Y * np.log(sig_value) + (1 - Y) * np.log(1 - sig_value)))/n # cost function
        def_cost = -1/n * np.dot(X.T, sig_value - Y) # differentition of cost
        beta = beta - learning_rate * def_cost
        if abs(prev_cost-cost) < tou:# if the difference is less than the threshold(tou) then break
            break
        prev_cost = cost
    return (beta, cost)

def lasso_logistic_regression(X, Y, epoch, learning_rate, tou, hyper_parameter):
    n = X.shape[0]
    m = X.shape[1]
    prev_cost = float('inf')
    beta = np.random.randn(m)

    for i in range(0, epoch):
        X_beta = X @ beta
        sig_value = 1 / (1 + np.exp(-X_beta))
        y = np.where(sig_value > 0.5, 1, 0)
        cost = -np.sum(Y * np.log(sig_value) + (1 - Y) * np.log(1 - sig_value)) / n + (hyper_parameter * np.sum(beta[1:n]))/n
        def_cost = -1/n * np.dot(X.T, sig_value - Y) + hyper_parameter/n
        
        beta = beta - learning_rate * def_cost
        if abs(prev_cost-cost) < tou:# if the difference is less than the threshold(tou) then break
            break
        prev_cost = cost
    return (beta, cost)

def ridge_logistic_regression(X, Y, epoch, learning_rate, tou, hyper_parameter):
    n = X.shape[0]
    beta = np.random.randn(len(X[0]))
    prev_cost = float('inf')
    for i in range(0, epoch):
        X_beta = X @ beta
        sig_value = 1 / (1 + np.exp(-X_beta))
        y = np.where(sig_value > 0.5, 1, 0)

        # regularization factor = (hyper_parameter * np.sum(abs(beta[1:n])
        cost = -np.sum(Y * np.log(sig_value) + (1 - Y) * np.log(1 - sig_value)) / n + (hyper_parameter * np.sum(beta[1:n] * beta[1:n]))/(2*n) # cost function

        def_cost = -1/n * np.dot(X.T, sig_value - Y) + (hyper_parameter * np.sum(beta[1:n]))/n
        beta = beta - learning_rate * def_cost
        if abs(prev_cost-cost) < tou: # if the difference is less than the threshold(tou) then break
            break
        prev_cost = cost
    return (beta, cost)

def accuracy(X_test_data, Y_test_data, beta): 
    x_b = np.dot(X_test_data, beta)
    sigmoid = 1 / (1 + np.exp(-x_b))
    y = np.where(sigmoid >= 0.5, 1, 0)
    result = np.equal(y, Y_test_data)
    return np.sum(result)/len(y) * 100

def cosine_similarity(beta, beta_predicted):
    cos_sim = np.dot(beta, beta_predicted) / (np.linalg.norm(beta) * np.linalg.norm(beta_predicted))
    return cos_sim

X, Y, beta = genrate_data(0.03, 5000, 2)
X_train_data, X_test_data = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
Y_train_data, Y_test_data = Y[:int(0.8 * len(X))], Y[int(0.8 * len(X)):]
print("original beta",beta)
print()
beta_predicted_logistic, cost_logistic = logistic_regression(X_train_data, Y_train_data, 500, 0.0001, 0.00001)
print("Accuracy Logistic Regression On trainig Data Data",accuracy(X_train_data, Y_train_data, beta_predicted_logistic))
print("Accuracy Logistic Regression On Test Data",accuracy(X_test_data, Y_test_data, beta_predicted_logistic))
print("Cost for Logistic Regression",cost_logistic)
print("predicted beta in logistic regression",beta_predicted_logistic)
print()

beta_predicted_lasso, cost_lasso = lasso_logistic_regression(X_train_data, Y_train_data, 500, 0.0001, 0.00001, 0.01)
print("Accuracy Lasso Regression On trainig Data Data",accuracy(X_train_data, Y_train_data, beta_predicted_lasso))
print("Accuracy Losso Regression On Test Data",accuracy(X_test_data, Y_test_data, beta_predicted_lasso))
print("Cost for Lasso Regression",cost_lasso)
print("predicted beta in Lasso Regression",beta_predicted_lasso)
print()

beta_predicted_ridge, cost_ridge = lasso_logistic_regression(X_train_data, Y_train_data, 500, 0.0001, 0.00001, 0.01)
print("Accuracy Ridge Regression On trainig Data Data",accuracy(X_train_data, Y_train_data, beta_predicted_ridge))
print("Accuracy Ridge Regression On Test Data",accuracy(X_test_data, Y_test_data, beta_predicted_ridge))
print("Cost for Ridge Regression",cost_ridge)
print("predicted beta in Ridge Regression",beta_predicted_ridge)





# cos = []
# n = []
# # cosine similarity and size of data
# for i in range(100, 10000, 100):
#     n.append(i)
#     X, Y, beta = genrate_data(0.01, i, 2)
#     X_train_data, X_test_data = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
#     Y_train_data, Y_test_data = Y[:int(0.8 * len(X))], Y[int(0.8 * len(X)):]
#     beta_predicted_logistic, cost_logistic = logistic_regression(X_train_data, Y_train_data, 500, 0.01, 0.000001)
#     cos_sim = cosine_similarity(beta, beta_predicted_logistic)
#     cos.append(cos_sim)
# plt.scatter(n, cos)
# plt.show()
# plt.plot(n,cos)
# plt.show()

# #cosine similarity and noise in data
# i = 0
# theta = 0.001
# while i < 50:
#     theta += 0.01
#     n.append(theta)
#     X, Y, beta = genrate_data(theta, 500, 2)
#     X_train_data, X_test_data = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
#     Y_train_data, Y_test_data = Y[:int(0.8 * len(X))], Y[int(0.8 * len(X)):]
#     beta_predicted_logistic, cost_logistic = logistic_regression(X_train_data, Y_train_data, 500, 0.01, 0.000001)
#     cos_sim = cosine_similarity(beta, beta_predicted_logistic)
#     cos.append(cos_sim)
#     theta += 0.01
#     i += 1
# plt.scatter(n, cos)
# plt.show()
# plt.plot(n,cos)
# plt.show()

