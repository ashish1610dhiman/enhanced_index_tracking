# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:42:45 2019

@author: Dhiman
"""
from mip import Model, xsum, maximize
import pandas as pd
import numpy as np
import random

""" Define the functions for Excess Return and Deviation Calculation """
def excess_return(returns,price,X_1,C):
    #import mip.model as mip
    z=[]
    T=returns.shape[0]
    for t in range(1,T+1):
        portfolio_return=[]
        for j in range(1,returns.shape[1]):
            r_jt=returns["security_{}".format(j)][t]
            q_jT=price["security_{}".format(j)][T]
            portfolio_return.append(r_jt*q_jT*(1/T)*X_1["security_{}".format(j)])
        benchmark_return=returns["index"][t]*C/T
        z.append(xsum(portfolio_return)-benchmark_return)
    return (xsum(z))

def deviation(price,returns,C,X_1,t):
    #import mip.model as mip
    theta=C/price["index"].iloc[-1]
    z=[] #For d
    for j in range(1,returns.shape[1]):
        q_jt=price["security_{}".format(j)][t]
        z.append(q_jt*X_1["security_{}".format(j)])
    return (theta*price["index"][t]-xsum(z))

def EIT_kernel(kernel,C,T,file,lamda,nuh,xii,k,pho,output):
    #from EIT_kernel import excess_return
    #from EIT_kernel import deviation
    """ Read the input index file """
    price=pd.read_csv("../input/index-weekly-data/index_{}.csv".format(file))
    price=price[["index"]+kernel][0:T+1]
    returns=(price-price.shift(1))/price.shift(1)
    returns.drop([0],axis=0,inplace=True)
    """ Parameters """
    tau=0 #Additional Cash Fund 
    #pho=0.4 #Transaction Cost Proportion
    c_b=0.01 #Constant for buying cost
    c_s=0.01 #Constant for selling cost
    f=min(price.min())/3 #Fixed Txn Cost
    """ Create the input variables """
    n=price.shape[1]-1
    X_0=np.zeros((n,1)) #Gives units of jth stock in original portfolio
    T=returns.shape[0] #Training limit
    theta=C/price["index"][T]
    for j in random.sample(kernel,k):
        X_0[kernel.index(j)]=(C/k)/price[j].iloc[0]
    """ Define the Linear Relaxation of EIT and necessary problem variables """
    #Solve EIT Kernel
    LP = Model("Linear Relaxation of EIT")
    #Gives units of jth stock in rebalaced portfolio
    X_1 = {x:LP.add_var(name="x1_{}".format(x),var_type="C",lb=0) for x in list(returns.columns)[1:]}
    #Binary Variable depicting if investor holds stock j
    y = {x:LP.add_var(name="y_{}".format(x),var_type="B") for x in list(returns.columns)[1:]}
    #Binary Variable depicting if stock j is traded
    w={x:LP.add_var(name="w_{}".format(x),var_type="B") for x in list(returns.columns)[1:]}
    #Buying cost of jth stock
    b= {x:LP.add_var(name="b_{}".format(x),var_type="C",lb=0) for x in list(returns.columns)[1:]}
    #Selling cost of jth stock
    s= {x:LP.add_var(name="s_{}".format(x),var_type="C",lb=0) for x in list(returns.columns)[1:]}
    #Downside Devaition
    d={x:LP.add_var(name="d_t{}".format(x),var_type="C",lb=0) for x in list(returns.index)}
    #Upside Devaition
    u={x:LP.add_var(name="u_t{}".format(x),var_type="C",lb=0) for x in list(returns.index)}
    """ Add Objective and Constraints """
    """ Objective """
    LP.objective=maximize(excess_return(returns,price,X_1,C))
    """ Constarints """
    stocks=list(returns.columns)[1:]  
    for stock in stocks:
        q_jT=price[stock][T]
        if X_0[kernel.index(stock)]==0:
            LP+=y[stock]==w[stock]
        #Constraint from eqn. 5
        LP+=(lamda*C*y[stock]<= X_1[stock]*q_jT)
        LP+=(X_1[stock]*q_jT <=nuh*C*y[stock])
        #Constraint from eqn. 8
        LP+=(b[stock]-s[stock]==(X_1[stock]*q_jT-float(X_0[kernel.index(stock)]*q_jT)))
        #Constraint from eqn. 9
        LP+=(b[stock]+s[stock]<=nuh*C*w[stock])
        #LP+=(b[stock]<=(nuh*C-X_0[j-1]*q_jT)*w[stock]) #Eqn 14
        #LP+=(s[stock]<=X_0[j-1]*q_jT*w[stock]) # Eqn 15
    #Constraint from eqn. 6
    LP+=(xsum(y.values())<=k)
    #Constraint from eqn. 7
    LP+=(xsum([X_1[stock]*price[stock][T] for stock in stocks])<=C)
    #Constraint from eqn. 10
    LP+=(xsum([c_b*b[stock]+c_s*s[stock]+f*w[stock] for stock in stocks])<=pho*C)
    for t in range(1,T+1):
        #Constraint from eqn. 4
        LP+=(d[t]-u[t]==deviation(price,returns,C,X_1,t))
    #Constraint from eqn. 16
    LP+=(xsum([d[t]+u[t] for t in range(1,T+1)])<=xii*C)
    print("Solving EIT(kernel)\n***************************************************")
    LP.emphasis=1
    status=LP.optimize()
    print("***************************************************\n")
    print("Optimisation Status={},Objective Value={}".format(str(status.value),str(round(LP.objective_value,3))))
    print("OPTIMAL(0), ERROR(-1), INFEASIBLE(1), UNBOUNDED(2)")
    result=pd.DataFrame()
    for stock in stocks:
        temp=pd.DataFrame()
        temp["security"]=[stock]
        temp["X_0"]=X_0[kernel.index(stock)]
        temp["X_1"]=[X_1[stock].x]
        temp["y"]=[y[stock].x]
        temp["w"]=[w[stock].x]
        temp["b"]=[b[stock].x]
        temp["s"]=[s[stock].x]
        result=result.append(temp,ignore_index=True)
    result.to_csv(output+"EIT_kernel_result_index_{}.csv".format(file),index=False)
    LP.write(output+"EIT_kernel_for_index_{}.lp".format(file))
    return(status,round(LP.objective_value,3))


   
def plot_results(kernel,result,file,T,output):
    import pandas as pd
    import numpy as np
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt
    price=pd.read_csv("../input/index-weekly-data/index_{}.csv".format(file))
    price=price[["index"]+kernel][0:T+1]
    returns=(price-price.shift(1))/price.shift(1)
    returns.drop([0],axis=0,inplace=True)
    q_T=price.iloc[T][1:]
    w=result["X_1"].values*q_T.values
    w=(w/np.sum(w))
    #Initialisation
    index=[1]
    tracking=[1]
    portfolio_return=[]
    #Read full 290 weeks data
    price=pd.read_csv("../input/index-weekly-data/index_{}.csv".format(file))
    price=price[["index"]+kernel]
    returns=(price-price.shift(1))/price.shift(1)
    returns.drop([0],axis=0,inplace=True)
    #Looping
    for t in returns.index:
        index.append((1+returns["index"][t])*index[-1])
        portfolio_return.append(sum(w*returns.loc[t][1:].values))
        tracking.append((1+portfolio_return[-1])*tracking[-1])
    #Plotting
    plot_df=pd.DataFrame()
    plot_df["index_value"]=index
    plot_df["portfolio_value"]=tracking
    plot_df["time_period"]=list(price.index)
    plot_df.index=price.index
    fig, ax = plt.subplots(figsize=(14,9))  
    ax.set_xlim(0,price.shape[0])
    try:
        ax.set_ylim(-0.3, 1.1*max(index+tracking))
    except:
        print ("Error in file={}".format(file))
    ind_1=plot_df[["time_period","index_value"]][0:T].values
    ind_2=plot_df[["time_period","index_value"]][T:].values
    port_1=plot_df[["time_period","portfolio_value"]][0:T].values
    port_2=plot_df[["time_period","portfolio_value"]][T:].values
    plt.plot(ind_1[:,0],ind_1[:,1],color=(57/255,62/255,68/255,0.7),label="Index")
    plt.plot(port_1[:,0],port_1[:,1],color=(255/255,87/255,86/255,0.43),label="Tracking Portfolio")
    plt.xlabel("Time in Weeks")
    plt.ylabel("Value of Index/Tracking Portfolio")
    ax.axvspan(T,price.shape[0],color=(57/255,62/255,68/255),alpha=0.025,label="Outside of Time")
    plt.axvline(x=T,color=(0,.20,.40))
    plt.legend(frameon=False,loc=2)
    cols=[(57/255,62/255,68/255,0.8),(255/255,87/255,86/255,0.8)]
    lc = LineCollection([ind_2,port_2],linewidths=(2,2),colors=cols,linestyles=["solid","solid"])
    ax.add_collection(lc)
    plt.fill_between(x=ind_2[:,0], y1=port_2[:,1]+3*np.std(portfolio_return[T:]),
              y2=port_2[:,1]-3*np.std(portfolio_return),
              color=(255/255,87/255,86/255,0.2))
    plt.title("EIT(kernel) performance for index={}\n".format(file))
    plt.savefig(output+"EIT_kernel_for_index_{}.jpg".format(file),dpi=250)