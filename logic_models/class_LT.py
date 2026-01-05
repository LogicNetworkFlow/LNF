import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gurobipy as gp
import re
from termcolor import colored
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

class logic_tree:

    def __init__(self, model, unique_node_dict, num_var_extra, TT, UU, VV, z_pi) -> None:

        self.m = model
        self.unique_node_dict = unique_node_dict
        self.dim_z = len(self.unique_node_dict.keys())
        self.ls_keys = list(unique_node_dict.keys())
        self.z_pi = z_pi
        
        for zz in self.z_pi:
            zz.vtype = gp.GRB.BINARY

        # stl_var does not need to be explicitly enforced as binary variables
        # stl_var = model.addMVar((num_var_extra, ), lb=0.0, ub=1.0, vtype=go.GRB.BINARY, name='stl_var')
        stl_var = model.addMVar((num_var_extra, ), lb=0.0, ub=1.0, name='stl_var')

        for ii in range(TT.shape[0]):
            tt_expr = gp.quicksum(TT[ii, j] * stl_var[j] for j in range(TT.shape[1]) if TT[ii, j] != 0)
            uu_expr = gp.quicksum(UU[ii, j] * self.z_pi[j] for j in range(UU.shape[1]) if UU[ii, j] != 0)
            model.addConstr(tt_expr + uu_expr <= VV[ii][0])
