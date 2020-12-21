"""
# Created by ashish1610dhiman at 19/12/20
Contact at ashish1610dhiman@gmail.com
"""

import os
import sys
import time
import pandas as pd
from collections import namedtuple

class TestEitDual:
    """
    Input: parameters for EIT_basic model
    Output: Result of EIT_basic model
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
