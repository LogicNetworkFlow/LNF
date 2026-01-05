import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from docplex.mp.model import Model as cplex_Model
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
        
        # Convert z_pi variables to binary for CPLEX
        for zz in self.z_pi:
            zz.set_vartype('B')  # Set as binary in DOcplex
        
        # Create STL variables
        stl_var = []
        for j in range(num_var_extra):
            var = model.continuous_var(lb=0.0, ub=1.0, name=f'stl_var_{j}')
            stl_var.append(var)

        for ii in range(TT.shape[0]):
            # Convert TT and UU rows to lists to avoid NumPy type errors
            TT_row = TT[ii, :].tolist()  # Convert NumPy row to list
            UU_row = UU[ii, :].tolist()  # Convert NumPy row to list
            
            # Explicitly sum the product of TT_row and stl_var
            lhs_tt_terms = []
            for jj in range(num_var_extra):
                if TT_row[jj] != 0:
                    lhs_tt_terms.append(TT_row[jj] * stl_var[jj])
            
            # Explicitly sum the product of UU_row and self.z_pi
            lhs_uu_terms = []
            for jj in range(self.dim_z):
                if UU_row[jj] != 0:
                    lhs_uu_terms.append(UU_row[jj] * self.z_pi[jj])
            
            # Combine all terms
            all_lhs_terms = lhs_tt_terms + lhs_uu_terms
            
            # Create LHS expression
            if len(all_lhs_terms) > 1:
                lhs_expr = model.sum(all_lhs_terms)
            elif len(all_lhs_terms) == 1:
                lhs_expr = all_lhs_terms[0]
            else:
                lhs_expr = 0  # No terms
            
            if isinstance(VV[ii], np.ndarray):
                rhs_value = VV[ii].item()
            elif isinstance(VV[ii], (list, tuple)):
                rhs_value = VV[ii][0]
            else:
                rhs_value = VV[ii]

            # Assert we have a proper scalar
            assert isinstance(rhs_value, (int, float, np.integer, np.floating)), f"RHS value must be a numeric scalar, got {type(rhs_value)}: {rhs_value}"
            
            # Add the constraint to the model
            model.add_constraint(lhs_expr <= rhs_value, ctname=f"logic_tree_constraint_{ii}")
            