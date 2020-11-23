import os
import sys
import time
import pandas as pd


class TestEitBasic:
    """
    Input: parameters for EIT_basic model
    Output: Result of EIT_basic model
    """

    def __init__(self, **kwargs):
        #print (kwargs)
        self.__dict__.update({param_name: param_name for  param_name, param_value in kwargs.items()})
        self.param_dict = kwargs
        self.best_objective_eit = None
        self.objective_linear = None

    def print_params(self):
        print(self.param_dict)

    def step_1(self, verbose=True):
        if verbose == True:
            print("+----------------------------------------------------+")
            print("    Step 1: Solving Linear Relaxation of EIT-Basic")
            print("+----------------------------------------------------+")
        s = time.time()
        # print ("len sys.argv in experiment={}".format(len(sys.argv)))
        out = os.system("python ./src/linear_relaxation.py {} {} {} {} {} {} {} {} {} {}". \
                        format(self.file, self.T, self.xii, self.k, self.pho, self.nuh, self.C,\
                               self.lamda, self.f, self.output))
        assert out == 0, "Error in linear relaxation"
        sys.exit(1)
        text_file = open(self.output + "/EIT_LP_details.txt")
        lines = text_file.readlines()
        failure = bool(int(lines[0][-2]))
        z_lp = float(lines[1].split("=")[-1][:-2])
        self.objective_linear = z_lp
        result_lp = pd.read_csv(self.output + "/result_index_{}.csv".format(self.file))
        if verbose == True:
            print("+----------------------------------------------------+")
            print("    Step 1 complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")
        return (failure, z_lp, result_lp)

    def step_2a(self, failure, z_lp, result_lp, verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2a: Sort Securities and create buckets")
            print("+----------------------------------------------------+")
        s = time.time()
        import src.sort_and_buckets
        # Create dummy problem using PULP
        LP, q_T = src.sort_and_buckets.dummy_problem(self.T, self.C, self.file)
        q_T.drop("index", inplace=True)
        objective = LP.objective

        # Create ranked list and buckets
        L = src.sort_and_buckets.sort_securities(result_lp, q_T, objective, self.lamda, self.C)  # Ranked List
        kernel = L[:self.m]
        initial_kernel = kernel.copy()  # Create copy of Initial Kernel
        buckets = src.sort_and_buckets.create_buckets(L, self.m, self.lbuck)
        Nb = len(buckets)
        return (kernel, buckets, L, Nb)
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2a complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")

    def step_2b(self, kernel, buckets, verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("     Step 2b: Solve EIT(kernel) and get lower-bound")
            print("+----------------------------------------------------+")
        s = time.time()
        import src.EIT_kernel
        try:
            status, z = src.EIT_kernel.EIT_kernel(kernel, self.C, self.T, self.file, self.lamda, \
                                                  self.nuh, self.xii, self.k, self.pho, self.f, self.output)
            failure = bool(status.value > 0)
        except:
            print("ERROR in EIT Kernel")
            failure =True
            print(status)

        execution_result = src.EIT_kernel.pd.DataFrame()
        temp = src.EIT_kernel.pd.DataFrame()
        temp["bucket"] = [0]
        temp["kernel_size"] = [len(kernel)]
        temp["problem_status"] = [status]
        temp["z_value"] = [z]
        execution_result = execution_result.append(temp, ignore_index=True)
        result_kernel = src.EIT_kernel.pd.read_csv(self.output + "/EIT_kernel_result_index_{}.csv".format(self.file))
        src.EIT_kernel.plot_results(kernel, result_kernel, self.file, self.T, self.output);
        return (z, failure)
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 2b complete in {:.2f}s".format(time.time() - s))
            print("+----------------------------------------------------+")

    def step_3(self,kernel, L, z, Nb, buckets, failure, verbose=True):
        if verbose:
            print("+----------------------------------------------------+")
            print("    Step 3: Execution Phase of Kernel Search")
            print("+----------------------------------------------------+")
        s = time.time()
        import src.EIT_bucket
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
                status, z, selected, EIT_model = src.EIT_bucket.EIT_bucket(kernel, bucket, i, failure, z_low, \
                                                                           self.C, self.T, self.file, self.lamda, self.nuh,
                                                                           self.xii, self.k, self.pho, self.f, self.output)
                eit_model_record.append(EIT_model)
            except:
                print("Error in this bucket")
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
                src.EIT_bucket.plot_results(kernel_copy, bucket, i, result_bucket, self.file, self.T, self.output)
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
            temp = src.EIT_bucket.pd.DataFrame()
            temp["bucket"] = [i]
            temp["kernel_size"] = [len(kernel)]
            temp["problem_status"] = [status]
            temp["z_value"] = [z]
            execution_result = execution_result.append(temp, ignore_index=True)
        self.best_objective_eit = z_low
        print("+----------------------------------------------------+")
        print("    Step 3 complete in {:.2f}s".format(time.time() - s))
        print("+----------------------------------------------------+")

    def print_result(self):
        print("+----------------------------------------------------+")
        print("BEST Objective from EIT:{:.3f}".format(self.best_objective_eit))
        print("BEST Objective from Linear Relaxation:{:.3f}".format(self.objective_linear))
        print("+----------------------------------------------------+")
