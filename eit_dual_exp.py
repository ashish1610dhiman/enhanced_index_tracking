"""
# Created by ashish1610dhiman at 19/12/20
Contact at ashish1610dhiman@gmail.com
"""

import os
import sys
import time
import pandas as pd
from collections import namedtuple
import traceback

class TestEitDual:
    """
    Input: parameters for EIT_dual model
    Output: Result of EIT_dual model
    """

    def __init__(self, **kwargs):
        #print (kwargs)
        self.__dict__.update({param_name: param_value for  param_name, param_value in kwargs.items()})
        self.param_dict = kwargs
        self.best_objective_eit = None
        self.objective_linear = None

    def print_params(self):
        print(self.param_dict)

    def give_params(self):
        params=namedtuple('params', [param_name for param_name in self.param_dict.keys()])
        my_params=params(**self.param_dict)
        return (my_params)

    def step_1(self, from_root=True,verbose=True):
        #print (os.getcwd())
        if verbose == True:
            print("+----------------------------------------------------+")
            print("    Step 1: Solving Linear Relaxation of EIT-Dual")
            print("+----------------------------------------------------+")
        s = time.time()
        # print ("len sys.argv in experiment={}".format(len(sys.argv)))
        path_root="." if from_root else ".."
        out = os.system("python {}/src_dual/linear_relaxation.py {} {} {} {} {} {} {} {} {} {} {} {} {}". \
                        format(path_root,self.file, self.T, self.xii, self.k, self.pho, self.nuh, self.C,\
                               self.lamda, self.f, self.w_return, self.w_risk, self.w_risk_down, self.output))
        #print (out)
        assert out == 0, "Error in linear relaxation"
        text_file = open(self.output + "/EIT_dual_LP_details.txt")
        lines = text_file.readlines()
        failure = bool(int(lines[0][-2]))
        if not failure:
            z_lp = float(lines[1].split("=")[-1][:-2])
            result_lp = pd.read_csv(self.output + "/result_eit_dual_index_{}.csv".format(self.file))
        else:
            z_lp = None
            result_lp = None
        self.objective_linear = z_lp
        if verbose == True:
            print("+----------------------------------------------------+")
            print("    Step 1 complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")
        return (failure, z_lp, result_lp)


    def step_2a(self, failure, z_lp, result_lp,from_root=True, verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2a: Sort Securities and create buckets")
            print("+----------------------------------------------------+")
        s = time.time()
        import src_dual.sort_and_buckets
        # Create dummy problem using PULP
        LP, q_T = src_dual.sort_and_buckets.dummy_problem(self.T, self.C, self.file, from_root, self.w_return,\
                                                          self.w_risk,self.w_risk_down, option=2)
        q_T.drop("index", inplace=True)
        objective = LP.objective

        # Create ranked list and buckets
        L = src_dual.sort_and_buckets.sort_securities(result_lp, q_T, objective, self.lamda, self.C)  # Ranked List
        kernel = L[:self.m]
        initial_kernel = kernel.copy()  # Create copy of Initial Kernel
        buckets = src_dual.sort_and_buckets.create_buckets(L, self.m, self.lbuck)
        Nb = len(buckets)
        return (kernel, buckets, L, Nb)
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2a complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")


    def step_2b(self, kernel, buckets,from_root, verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("     Step 2b: Solve EIT(kernel) and get lower-bound")
            print("+----------------------------------------------------+")
        s = time.time()
        import src_dual.EIT_kernel
        try:
            status, z, in_excess_return, in_tr_err, out_excess_return, out_tr_err, portfolio_size = \
                src_dual.EIT_kernel.EIT_kernel(kernel, self.C, self.T, self.file, self.lamda, self.nuh, self.xii, \
                                          self.k, self.pho, self.f, self.output, self.w_return, self.w_risk, \
                                          self.w_risk_down, from_root)
            failure = bool(status.value > 0)
        except Exception as e:
            print("ERROR in EIT Kernel")
            traceback.print_exc()
            failure = True

        execution_result = src_dual.EIT_kernel.pd.DataFrame()
        temp = src_dual.EIT_kernel.pd.DataFrame()
        temp["bucket"] = [0]
        temp["kernel_size"] = [len(kernel)]
        temp["problem_status"] = [status]
        temp["z_value"] = [z]
        temp["in_excess_return"]=[in_excess_return]
        temp["in_tr_err"]=[in_tr_err]
        temp["out_excess_return"]=[out_excess_return]
        temp["out_tr_err"]=[out_tr_err]
        temp["portfolio_size"] = [portfolio_size]
        execution_result = execution_result.append(temp, ignore_index=True)
        result_kernel = src_dual.EIT_kernel.pd.read_csv(self.output + \
                                                        "/EIT_dual_kernel_result_index_{}.csv".format(self.file))
        src_dual.EIT_kernel.plot_results(kernel, result_kernel, self.file, self.T, self.output,from_root);
        return (z, failure,execution_result)
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2b complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")

    def step_3(self, kernel, L, z, Nb, buckets, failure, execution_result, verbose=True, from_root=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 3: Execution Phase of Kernel Search")
            print("+----------------------------------------------------+")
        s = time.time()
        import src_dual.EIT_bucket
        """Initialise p_dict and z_low"""
        p_dict = {}
        for stock in L:
            p_dict[stock] = 0
        z_low = z

        kernel_record = []
        eit_model_record = []
        for i in range(1, Nb + 1):
            bucket = buckets[i]
            # Add bucket to kernel
            kernel_copy = kernel.copy()
            kernel_record.append(kernel_copy)
            # kernel=kernel+bucket
            print("\n\nFor bucket={}".format(str(i)))
            # Solve EIT(K+Bi)
            try:
                status, z, selected, EIT_model, in_excess_return,\
                in_tr_err, out_excess_return, out_tr_err, portfolio_size = \
                    src_dual.EIT_bucket.EIT_bucket(kernel, bucket, i, failure, z_low,self.C, self.T, self.file,\
                                              self.lamda, self.nuh, self.xii, self.k, self.pho, self.f,\
                                              self.output, self.w_return, self.w_risk, self. w_risk_down, from_root)
                eit_model_record.append(EIT_model)
            except Exception as e:
                print("Error in this bucket")
                traceback.print_exc()
                continue
            if status.value == 0:  # Check if EIT(kernel+bucket) is feasible
                if failure == True:  # check if EIT(Kernel) was in-feasible
                    failure = False
                # Update lower_bound
                print("Updating Lower Bound")
                if (z > z_low):
                    z_low = z
                """Update Kernel"""
                # Add stocks from bucket which are selected in optimal
                print("Updating Kernel")
                print("Length of Old Kernel={}".format(len(kernel_copy)))
                kernel = kernel_copy + selected
                print("Length of Updated Kernel={}".format(len(kernel)))
                # Make p=0 if stock just selected in Kernel
                for stock in selected:
                    p_dict[stock] = 0
                # Update p_dict
                result_bucket = pd.read_csv(self.output + "/EIT_bucket_{}_result_index_{}.csv".format(i, self.file))
                src_dual.EIT_bucket.plot_results(kernel_copy, bucket, i, result_bucket, self.file,\
                                            self.T, self.output,from_root)
                result_bucket.index = result_bucket["security"]
                result_bucket.drop(["security"], axis=1, inplace=True)
                for stock in kernel:
                    if (result_bucket.loc[stock]['y'] == 0):
                        p_dict[stock] += 1  # Increase by 1 if not selected in optimal
                # Remove from Kernel
                to_remove = [stock for (stock, p_value) in p_dict.items() if p_value > self.p]
                for stock in to_remove:
                    if stock in kernel:
                        print("Removing {} from kernel".format(stock))
                        kernel.remove(stock)
                        p_dict[stock] = 0
                print("Current Length Kernel={}".format(len(kernel)))
            else:
                kernel = kernel_copy
            temp = src_dual.EIT_bucket.pd.DataFrame()
            temp["bucket"] = [i]
            temp["kernel_size"] = [len(kernel)]
            temp["problem_status"] = [status]
            temp["z_value"] = [z]
            temp["in_excess_return"] = [in_excess_return]
            temp["in_tr_err"] = [in_tr_err]
            temp["out_excess_return"] = [out_excess_return]
            temp["out_tr_err"] = [out_tr_err]
            temp["portfolio_size"] = [portfolio_size]
            execution_result = execution_result.append(temp, ignore_index=True)
        self.best_objective_eit = z_low
        return (execution_result)
        print("+----------------------------------------------------+")
        print("    Step 3 complete in {:.2f}s".format(time.time() - s))
        print("+----------------------------------------------------+")

    def print_result(self):
        if (self.objective_linear !=None) and (self.best_objective_eit!=None):
            print("+----------------------------------------------------+")
            print("BEST Objective from EIT:{:.3f}".format(self.best_objective_eit))
            print("BEST Objective from Linear Relaxation:{:.3f}".format(self.objective_linear))
            print("+----------------------------------------------------+")
        else:
            print ("either of 2 None")


    def run_experiment(self,from_root=True,verbose=True):
        failure, z_lp, result_lp = self.step_1(from_root=from_root,verbose=verbose)
        if not failure:
            kernel, buckets, L, Nb = self.step_2a(failure, z_lp, result_lp,from_root=from_root,verbose=verbose)
            z, failure,execution_result = self.step_2b(kernel, buckets, from_root=from_root, verbose=verbose)
            if not failure:
                execution_result=self.step_3(kernel, L, z, Nb, buckets, failure,execution_result,\
                                             from_root=from_root,verbose=verbose)
            else:
                execution_result = None
        else:
            execution_result=None
        if verbose:
            self.print_result()
        return (execution_result)

