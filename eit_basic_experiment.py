import os
import sys
import time
import pandas as pd


class test_eit_basic:
    """
    Input: parameters for EIT_basic model
    Output: Result of EIT_basic model
    """
    def __init__(self,*kwargs):
        for param_name,param_value in kwargs.items():
            self.param_name=param_value
        self.param_dict=kwargs



    def print_params(self):
        print (self.param_dict)



    def step_1(self,verbose=True):
        if verbose ==True:
            print("+----------------------------------------------------+")
            print("    Step 1: Solving Linear Relaxation of EIT-Basic")
            print("+----------------------------------------------------+")
        s = time.time()
        # print ("len sys.argv in experiment={}".format(len(sys.argv)))
        out = os.system("python ./src/linear_relaxation.py {} {} {} {} {} {} {} {} {} {}".\
                        format(self.file, self.T, self.xii, self.k, self.pho, self.nuh,\
                               self.C, self.lamda, self.f, self.output))
        assert out ==0, "Error in linear relaxation"
        text_file = open(self.output + "/EIT_LP_details.txt")
        lines = text_file.readlines()
        failure = bool(int(lines[0][-2]))
        z_lp = float(lines[1].split("=")[-1][:-2])
        result_lp = pd.read_csv(self.output + "/result_index_{}.csv".format(self.file))
        if verbose == True:
            print("+----------------------------------------------------+")
            print("    Step 1 complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")
        return (failure,z_lp,result_lp)



    def step_2a(self,failure,z_lp,result_lp,verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2a: Sort Securities and create bucktes")
            print("+----------------------------------------------------+")
        s = time.time()
        from src.sort_and_buckets import *
        # Create dummy problem using PULP
        LP, q_T = dummy_problem(self.T, self.C, self.file)
        q_T.drop("index", inplace=True)
        objective = LP.objective

        # Create ranked list and buckets
        L = sort_securities(result_lp, q_T, objective, self.lamda, self.C)  # Ranked List
        kernel = L[:m]
        initial_kernel = kernel.copy()  # Create copy of Initial Kernel
        buckets = create_buckets(L, self.m, self.lbuck)
        Nb = len(buckets)
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2a complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")



    def step_2b(self,verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2b: Sort Securities and create bucktes")
            print("+----------------------------------------------------+")
        s = time.time()
        from src.EIT_kernel import *
        try:
            status, z = EIT_kernel(kernel, C, T, file, lamda, nuh, xii, k, pho, f, output)
            failure = bool(status.value > 0)
        except:
            print("ERROR in EIT Kernel")
            print(status)

        execution_result = pd.DataFrame()
        temp = pd.DataFrame()
        temp["bucket"] = [0]
        temp["kernel_size"] = [len(kernel)]
        temp["problem_status"] = [status]
        temp["z_value"] = [z]
        execution_result = execution_result.append(temp, ignore_index=True)
        result_kernel = pd.read_csv(output + "/EIT_kernel_result_index_{}.csv".format(file))
        plot_results(kernel, result_kernel, file, T, output);
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2b complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")


#
# """ Read paramaters as CMD line arguments """
# print ("Running Linear Relaxation of EIT")
# #assert __name__ == "__main__", "Error, not designed to run as module"
# if len(sys.argv)!=11:
#     print("Error, Wrong no. of arguments={}, using default arguments".format(len(sys.argv)))
#     file=1
#     T=200
#     xii=0.8
#     k=12
#     pho=0.4
#     nuh=0.65
#     C = 1000000
#     lamda = 1 / (100 * C)
#     f=12
#     output="./experiment_1/"
#     #sys.exit(1)
# else:
#     #print (sys.argv[1])
#     file=int(sys.argv[1])
#     T=int(sys.argv[2])
#     xii=float(sys.argv[3]) #Proportion cosntant for TrE
#     k=int(sys.argv[4])
#     pho=float(sys.argv[5])
#     nuh = float(sys.argv[6])
#     C=float(sys.argv[7])
#     lamda=float(sys.argv[8])
#     f=float(sys.argv[9])
#     output="./"+str(sys.argv[10])+"/"
#
#
# print ("+----------------------------------------------------+")
# print ("Step 1: Solving Linear Relaxation of EIT-Basic")
# print ("+----------------------------------------------------+")
# s=time.time()
# #print ("len sys.argv in experiment={}".format(len(sys.argv)))
# os.system("python ./src/linear_relaxation.py {} {} {} {} {} {} {} {} {} {}".format(\
#     file,T,xii,k,pho,nuh,C,lamda,f,output))
#
# text_file=open(output+"/EIT_LP_details.txt")
# lines=text_file.readlines()
# failure=bool(int(lines[0][-2]))
# z_lp=float(lines[1].split("=")[-1][:-2])
# result_lp=pd.read_csv(output+"/result_index_{}.csv".format(file))
# print ("+----------------------------------------------------+")
# print ("Step 1 complete in {:.2f}s".format(time.time()-s))
# print ("+----------------------------------------------------+")
#
#
# print ("+----------------------------------------------------+")
# print ("Step 2a: Sort Securities and create bucktes")
# print ("+----------------------------------------------------+")
# s=time.time()
# # Create dummy problem using PULP
# LP,q_T=dummy_problem(T,C,file)
# q_T.drop("index",inplace=True)
# objective=LP.objective
#
# #Create ranked list and buckets
# L=sort_securities(result_lp,q_T,objective,lamda,C) #Ranked List
# kernel=L[:m]
# initial_kernel=kernel.copy() #Create copy of Initial Kernel
# buckets=create_buckets(L,m,lbuck)
# Nb=len(buckets)
# print ("+----------------------------------------------------+")
# print ("Step 2a complete in {:.2f}s".format(time.time()-s))
# print ("+----------------------------------------------------+")