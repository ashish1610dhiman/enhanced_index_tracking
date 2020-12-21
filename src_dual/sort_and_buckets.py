# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:15:32 2019

@author: Dhiman
"""

import pulp as pulp
import os
import pandas as pd
import numpy as np


def reduced_cost(objective):
    cj = pd.DataFrame()
    security_list = []
    cost_list = []
    for key, value in objective.items():
        security_list.append(key.name[3:])
        cost_list.append(value)
    cj["cost"] = cost_list
    cj.index = security_list
    return cj


def sort_securities(result, q_T, objective, lamda, C):
    value = pd.DataFrame()
    value["value"] = result["X_1"].values * q_T.values
    value.index = q_T.index
    value.sort_values(by="value", ascending=False, inplace=True)
    not_optimal = value[value["value"] == 0]
    cj = reduced_cost(objective)
    for security in not_optimal.index:
        not_optimal.loc[security]['value'] = cj.loc[security] * (lamda * C) / q_T[security]
    not_optimal.sort_values(by="value", ascending=False, inplace=True)
    sorted_securities = list(value.index[value["value"] > 0]) + list(not_optimal.index)
    return (sorted_securities)


def excess_return(returns, price, X_1, C):
    # import pulp as pulp
    T = returns.shape[0]
    z = []
    theta = C / price["index"].iloc[-1]
    for t in range(1, T + 1):
        portfolio_return = []
        for j in range(1, returns.shape[1]):
            r_jt = returns["security_{}".format(j)][t]
            q_jT = price["security_{}".format(j)][T]
            portfolio_return.append(r_jt * q_jT * X_1["security_{}".format(j)])
        benchmark_return = returns["index"][t] * C
        z.append((pulp.lpSum(portfolio_return) - benchmark_return) / (theta * price["index"][t]))
    return (pulp.lpSum(z) / T)


def deviation(price, returns, C, X_1, t):
    # import pulp as pulp
    theta = C / price["index"].iloc[-1]
    z = []  # For d
    for j in range(1, returns.shape[1]):
        q_jt = price["security_{}".format(j)][t]
        z.append(q_jt * X_1["security_{}".format(j)])
    return ((theta * price["index"][t] - pulp.lpSum(z) )/ (theta * price["index"][t]))


def create_buckets(L, m, lbuck):
    buckets = {}
    Nb = int((len(L) - m) / lbuck) + 1
    i = 1
    a = m
    while (i <= Nb):
        b = a + lbuck
        if (b < len(L)):
            buckets[i] = L[a:b]
        else:
            buckets[i] = L[a:]
        a = b
        i = i + 1
    return buckets


def dummy_problem(T, C, file, from_root, w_return, w_risk, option):
    # import pulp as pulp
    # from sort_and_buckets import excess_return
    # from sort_and_buckets import deviation
    if from_root:
        file_path = "./input/index-weekly-data/index_{}.csv"
    else:
        n_dirs_up = os.getcwd().split("/")
        n_dirs_up.reverse()
        n_dirs_up = n_dirs_up.index("eit_paper")
        root_path = "/".join([".."] * n_dirs_up)
        file_path = root_path + "/input/index-weekly-data/index_{}.csv"
    price = pd.read_csv(file_path.format(file))
    price = price[0:T + 1]
    returns = (price - price.shift(1)) / price.shift(1)
    returns.drop([0], axis=0, inplace=True)
    LP = pulp.LpProblem("Linear Relaxation of EIT_dual", pulp.LpMaximize)
    X_1 = pulp.LpVariable.dict('x1_%s', list(returns.columns)[1:], lowBound=0)
    d = pulp.LpVariable.dict('d_t%s', list(returns.index), lowBound=0)
    u = pulp.LpVariable.dict('u_t%s', list(returns.index), lowBound=0)
    exc_ret = excess_return(returns, price, X_1, C)
    if option == 1:
        LP += w_return * (exc_ret)
    if option == 2:
        LP += w_return * (exc_ret) - \
              w_risk * (pulp.lpSum([(d[t] + u[t])/(price["index"][t])  for t in range(1, T + 1)]))
        for t in range(1, T + 1):
            LP += (d[t] - u[t] == deviation(price, returns, C, X_1, t))
    q_T = price.iloc[T]
    return (LP, q_T)
