# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:54:42 2019

@author: Dhiman
"""

import os
import pandas as pd
import numpy as np
import random
from mip import Model, xsum, maximize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import sys

""" Define the functions for Excess Return and Deviation Calculation """


def excess_return(returns, price, X_1, C):
    # import mip.model as mip
    z = []
    T = returns.shape[0]
    for t in range(1, T + 1):
        portfolio_return = []
        for j in range(1, returns.shape[1]):
            r_jt = returns["security_{}".format(j)][t]
            q_jT = price["security_{}".format(j)][T]
            portfolio_return.append(r_jt * q_jT * (1 / T) * X_1["security_{}".format(j)])
        benchmark_return = returns["index"][t] * C / T
        z.append(xsum(portfolio_return) - benchmark_return)
    return (xsum(z))


# util test func
def excess_return_val(returns, price, X_1, C):
    # import mip.model as mip
    z = []
    T = returns.shape[0]
    for t in range(1, T + 1):
        portfolio_return = []
        for j in range(1, returns.shape[1]):
            r_jt = returns["security_{}".format(j)][t]
            q_jT = price["security_{}".format(j)][T]
            portfolio_return.append(r_jt * q_jT * (1 / T) * X_1["security_{}".format(j)].x)
        benchmark_return = returns["index"][t] * C / T
        z.append(sum(portfolio_return) - benchmark_return)
    return (sum(z))


def deviation(price, returns, C, X_1, t):
    # import mip.model as mip
    theta = C / price["index"].iloc[-1]
    z = []  # For d
    for j in range(1, returns.shape[1]):
        q_jt = price["security_{}".format(j)][t]
        z.append(q_jt * X_1["security_{}".format(j)])
    return (theta * price["index"][t] - xsum(z))


""" Read CMD line arguments """
print("Running Linear Relaxation of EIT ...")
# print ("len sys.argv in linear relax={}".format(len(sys.argv)))
if len(sys.argv) != 11:
    print("Error, Wrong no. of arguments={}, using default arguments".format(len(sys.argv)))
    file = 1
    T = 200
    xii = 0.8
    k = 12
    pho = 0.4
    nuh = 0.65
    C = 1000000
    lamda = 1 / (100 * C)
    f = 12
    output = "./experiment_1/"
    # sys.exit(1)
else:
    # print (sys.argv[1])
    file = sys.argv[1]
    T = int(sys.argv[2])
    xii = float(sys.argv[3])  # Proportion cosntant for TrE
    k = int(sys.argv[4])
    pho = float(sys.argv[5])
    nuh = float(sys.argv[6])
    C = float(sys.argv[7])
    lamda = float(sys.argv[8])
    f = float(sys.argv[9])
    output = "./" + str(sys.argv[10]) + "/"

if not os.path.exists(output):
    os.makedirs(output)

print(os.getcwd())
if os.getcwd().split("/")[-1] == "eit_paper":
    file_path = "./input/index-weekly-data/index_{}.csv"
else:
    n_dirs_up = os.getcwd().split("/")
    n_dirs_up.reverse()
    n_dirs_up = n_dirs_up.index("eit_paper")
    root_path = "/".join([".."] * n_dirs_up)
    file_path = root_path + "/input/index-weekly-data/index_{}.csv"
""" Read the input index file """
price = pd.read_csv(file_path.format(file))
price = price[0:T + 1]
returns = (price - price.shift(1)) / price.shift(1)
returns.drop([0], axis=0, inplace=True)

""" Define Parameters of the model and create input vars """
# C=1000000 #Capital available
tau = 0  # Additional Cash Fund
# lamda=1/(100*C) # lower_bound for capital invested in jth stock
# nuh=0.65  # upper_bound
# k= 12 #Cardinality Constraint
# pho=0.4 #Transaction Cost Proportion
c_b = 0.01  # Constant for buying cost
c_s = 0.01  # Constant for selling cost
# f=min(price.min())/3 #Fixed Txn Cost
# xii=0.5

""" Create the input variables """
n = price.shape[1] - 1
X_0 = np.zeros((n, 1))  # Gives units of jth stock in original portfolio
T = returns.shape[0]  # Training limit
theta = C / price["index"][T]
for j in random.sample(range(1, n + 1), k):
    # print (j)
    X_0[j - 1] = (C / k) / price["security_{}".format(j)].iloc[0]

""" Define the Linear Relaxation of EIT and necessary problem variables """
print("Solving LP(EIT)\n***************************************************")
# Solve LP Relaxation
LP = Model("Linear Relaxation of EIT")

# Gives units of jth stock in rebalaced portfolio
X_1 = {x: LP.add_var(name="x1_{}".format(x), var_type="C", lb=0) for x in list(returns.columns)[1:]}
# Binary Variable depicting if investor holds stock j
y = {x: LP.add_var(name="y_{}".format(x), var_type="C", lb=0, ub=1) for x in list(returns.columns)[1:]}
# Binary Variable depicting if stock j is traded
w = {x: LP.add_var(name="w_{}".format(x), var_type="C", lb=0, ub=1) for x in list(returns.columns)[1:]}
# Buying cost of jth stock
b = {x: LP.add_var(name="b_{}".format(x), var_type="C", lb=0) for x in list(returns.columns)[1:]}
# Selling cost of jth stock
s = {x: LP.add_var(name="s_{}".format(x), var_type="C", lb=0) for x in list(returns.columns)[1:]}
# Downside Devaition
d = {x: LP.add_var(name="d_t{}".format(x), var_type="C", lb=0) for x in list(returns.index)}
# Upside Devaition
u = {x: LP.add_var(name="u_t{}".format(x), var_type="C", lb=0) for x in list(returns.index)}

""" Add Objective and Constraints """
""" Objective """
LP.objective = maximize(excess_return(returns, price, X_1, C))

""" Constarints """
for j in range(1, returns.shape[1]):
    stock = "security_{}".format(j)
    q_jT = price[stock][T]
    if X_0[j - 1] == 0:
        LP += y[stock] == w[stock]
    # Constraint from eqn. 5
    LP += (lamda * C * y[stock] <= X_1[stock] * q_jT)
    LP += (X_1[stock] * q_jT <= nuh * C * y[stock])
    # Constraint from eqn. 8
    LP += (b[stock] - s[stock] == (X_1[stock] * q_jT - float(X_0[j - 1] * q_jT)))
    # Constraint from eqn. 9
    LP += (b[stock] + s[stock] <= nuh * C * w[stock])
    # LP+=(b[stock]<=(nuh*C-X_0[j-1]*q_jT)*w[stock]) #Eqn 14
    # LP+=(s[stock]<=X_0[j-1]*q_jT*w[stock]) # Eqn 15

# Constraint from eqn. 6
LP += (xsum(y.values()) <= k)

stocks = ["security_{}".format(j) for j in range(1, returns.shape[1])]
# Constraint from eqn. 7
LP += (xsum([X_1[stock] * price[stock][T] for stock in stocks]) == C)

# Constraint from eqn. 10
LP += (xsum([c_b * b[stock] + c_s * s[stock] + f * w[stock] for stock in stocks]) <= pho * C)

for t in range(1, T + 1):
    # Constraint from eqn. 4
    LP += (d[t] - u[t] == deviation(price, returns, C, X_1, t))

# Constraint from eqn. 16
LP += (xsum([d[t] + u[t] for t in range(1, T + 1)]) <= xii * C)

""" Solve the problem and collect the results """
status = LP.optimize()
print("***************************************************\n")
print("Optimisation Status={}".format(str(status.value)))
print("OPTIMAL(0), ERROR(-1), INFEASIBLE(1), UNBOUNDED(2)")
result = pd.DataFrame()
for stock in stocks:
    temp = pd.DataFrame()
    temp["security"] = [stock]
    temp["X_0"] = X_0[int(stock.split('_')[-1]) - 1]
    temp["X_1"] = [X_1[stock].x]
    temp["y"] = [y[stock].x]
    temp["w"] = [w[stock].x]
    temp["b"] = [b[stock].x]
    temp["s"] = [s[stock].x]
    result = result.append(temp, ignore_index=True)

# result.to_csv(output+"result_index_{}.csv".format(file),index=False)
file1 = open(output + "EIT_LP_details.txt", "w+")
file1.writelines("LP(EIT) status={}\nObjective={}\n".format(str(LP.status.value), LP.objective_value))
file1.writelines("xii={},k={},lambda={},nuh={}".format(xii, k, lamda, nuh))
file1.close()

# Write LP model to a file
LP.write(output + "/LP_EIT_index_{}.lp".format(file))

if LP.status.value != 0:
    sys.exit(0)

# Calulation
q_T = price.iloc[T][1:]
w = result["X_1"].values * q_T.values
w = (w / np.sum(w))
result["weights"] = w
# =q_T.reset_index()
result["q_T"] = q_T.values
result.to_csv(output + "/result_index_{}.csv".format(file), index=False)
# Initialisation
index = [1]
tracking = [1]
portfolio_return = []
# Read full 290 weeks data
price = pd.read_csv(file_path.format(file))
returns = (price - price.shift(1)) / price.shift(1)
returns.drop([0], axis=0, inplace=True)
# Looping
for t in returns.index:
    index.append((1 + returns["index"][t]) * index[-1])
    portfolio_return.append(sum(w * returns.loc[t][1:].values))
    tracking.append((1 + portfolio_return[-1]) * tracking[-1])
# Plotting
plot_df = pd.DataFrame()
plot_df["index_value"] = index
plot_df["portfolio_value"] = tracking
plot_df["time_period"] = list(price.index)
plot_df.index = price.index
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, price.shape[0])
try:
    ax.set_ylim(-0.3, 1.1 * max(index + tracking))
except:
    print("Error in file={}".format(file))
ind_1 = plot_df[["time_period", "index_value"]][0:T].values
ind_2 = plot_df[["time_period", "index_value"]][T:].values
port_1 = plot_df[["time_period", "portfolio_value"]][0:T].values
port_2 = plot_df[["time_period", "portfolio_value"]][T:].values
plt.plot(ind_1[:, 0], ind_1[:, 1], color=(57 / 255, 62 / 255, 68 / 255, 0.7), label="Index")
plt.plot(port_1[:, 0], port_1[:, 1], color=(255 / 255, 87 / 255, 86 / 255, 0.43), label="Tracking Portfolio")
plt.xlabel("Time in Weeks")
plt.ylabel("Value of Index/Tracking Portfolio")
ax.axvspan(T, price.shape[0], color=(57 / 255, 62 / 255, 68 / 255), alpha=0.025, label="Outside of Time")
plt.axvline(x=T, color=(0, .20, .40))
plt.legend(frameon=False, loc=2)
cols = [(57 / 255, 62 / 255, 68 / 255, 0.8), (255 / 255, 87 / 255, 86 / 255, 0.8)]
lc = LineCollection([ind_2, port_2], linewidths=(2, 2), colors=cols, linestyles=["solid", "solid"])
ax.add_collection(lc)
plt.fill_between(x=ind_2[:, 0], y1=port_2[:, 1] + 3 * np.std(portfolio_return[T:]),
                 y2=port_2[:, 1] - 3 * np.std(portfolio_return),
                 color=(255 / 255, 87 / 255, 86 / 255, 0.2))
plt.title("Linear Relaxation of EIT for index={}\n".format(file))
plt.suptitle("\nxii={},k={},lambda={},nuh={}".format(xii, k, lamda, nuh), fontsize=11)
plt.savefig(output + "LP_EIT for index_{}.jpg".format(file), dpi=250)
