## Results for EIT-Basic vs EIT-Dual
metric = Excess Return/ Tracking Error  
Excess-Return = Average over time[(return on portfolio)_t−(return on index)_t]  
Tracking-Error = Average over time[| (index value)_t−(portfolio value)_t |]  

### Instance:1
Hang Seng | 31 stocks | March 1992 to September 1997
> ####  Instance:1a 
> un-reduced dataset | weekly data | Training = 200 | OOS = 91  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.05,0.1,0.2]  
>> xii in [1.2,1.3,1.4]  
>> k in [12,16,25]  
>> m in [8,12,16]  
>> nuh in [0.3,0.45,0.6]  
>> w_return=100, w_risk=110, w_risk_down=1  
> ---------------------- ********************** ----------------------------
> ##### In-sample metric:
>> EIT-Dual beats old approach 66.6% times
> ##### OOS metric:
>> EIT-Dual beats old approach 92.6% times


> ####  Instance:1b 
> NPCA Reduced | bi-weekly data | Training = 42 | OOS = 17  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.05,0.1,0.2]  
>> xii in [0.3,0.4,0.5]  
>> k in [12,16,25]  
>> m in [8,12,16] 
>> nuh in [0.2,0.25,0.3]  
>> w_return=10 , w_risk=1100000, w_risk_down=10  
> ---------------------- ********************** ----------------------------
> ##### In-sample metric:
>> EIT-Dual beats old approach 66.6% times
> ##### OOS metric:
>> EIT-Dual beats old approach 55.5% times


> ####  Instance:1c 
> NMF Reduced | bi-weekly data | Training = 42 | OOS = 17  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.05,0.1,0.2]  
>> xii in [1.2,1.3,1.4]  
>> k in [12,16,25]  
>> m in [8,12,16]  
>> nuh in [0.3,0.45,0.6] 
>> w_return=100, w_risk=11000, w_risk_down=10    
> ---------------------- ********************** ----------------------------
> * EIT-Dual beats old approach 0 times on in-sample metric.
> * EIT-Dual beats old approach 74.1% times on in-sample metric.



### Instance:2
S&P500 | 500 stocks | Feb 2013 to Mar 2018
> ####  Instance:2a 
> un-reduced dataset | daily data | Training = 880 | OOS = 218  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.2,0.3]  
>> xii in [1.1,1.2]  
>> k in [100,120,140]  
>> m in [70,90]  
>> nuh in [0.3,0.35,0.4]     
> ---------------------- ********************** ----------------------------
> * EIT-Dual beats old approach ??? times on in-sample metric.
> * EIT-Dual beats old approach ??? times on in-sample metric.
> * Taking too long too solve

 
> ####  Instance:2b 
> UnReduced dataset | weekly data | Training = 180 | OOS = 57  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.2]  
>> xii in [1.1]  
>> k in [100]  
>> m in [50]  
>> nuh in [0.3,0.35,0.4]     
> ---------------------- ********************** ----------------------------
> ##### In-sample metric:
>> EIT-Dual beats old approach 100% times
> ##### OOS metric:
>> EIT-Dual beats old approach 33% times | can be improved by altering w_risk


> ####  Instance:2c TODO
> NPCA Reduced | weekly data | Training = 200 | OOS = 97  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.25]  
>> xii in [1.1]  
>> k in [70,80]  
>> m in [50]  
>> nuh in [0.15,0.2,0.25]     
> ---------------------- ********************** ----------------------------
> * EIT-Dual beats old approach 100% times on in-sample metric.
> * EIT-Dual beats old approach 33% times on in-sample metric.

 
> ####  Instance:2d TODO
> NMF Reduced | weekly data | Training = 200 | OOS = 97  
> ---------------------- Parameter Combinations ----------------------------  
>> pho in [0.25]  
>> xii in [1.1]  
>> k in [70,80]  
>> m in [50]  
>> nuh in [0.15,0.2,0.25]       
> ---------------------- ********************** ----------------------------
> * EIT-Dual beats old approach 100% times on in-sample metric.
> * EIT-Dual beats old approach 100% times on in-sample metric.
