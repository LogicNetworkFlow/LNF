import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gurobipy as go
from termcolor import colored
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

class logic_nf:
    def __init__(self, model, unique_predicate_dict, node_list, edge_list, output, z_pi) -> None:
        self.m = model
        self.unique_predicate_dict = unique_predicate_dict
        self.node_list = node_list
        self.edge_list = edge_list
        self.output = output
        self.num_edges = len(self.edge_list)
        self.dim_z = len(self.unique_predicate_dict.keys())
        self.ls_keys = list(unique_predicate_dict.keys())                
        self.z_pi = z_pi
        self.inf = 1.0

        for zz in self.z_pi:
            zz.vtype = go.GRB.BINARY

        # Create single shared ye_logic
        # ye_logic can be set to non-binary variale, the solution will automatically be either 0 or 1
        # self.ye_logic = self.m.addMVar(self.num_edges, lb=0.0, ub=1.0, name='ye_logic', vtype = go.GRB.BINARY)
        self.ye_logic = self.m.addMVar(self.num_edges, lb=0.0, ub=1.0, name='ye_logic'); print(colored('ye_logic are not explicitly enforced as binary variables!', 'red'))

        print("Setting up combined flow structure with lazy variable creation...")
        self.enforce_flow_structure()
        # print("Setting objectives...")
        # self.set_objective()
        self.m.update()

    def enforce_flow_structure(self):

        # Enforce shared ye_logic flow conservation - the structure of y is unchanged
        for nn in self.node_list:
            if nn['in'] != [] and nn['out'] != []:
                y_in = sum(self.ye_logic[inn].item() for inn in nn['in'])
                y_out = sum(self.ye_logic[ouu].item() for ouu in nn['out'])
                self.m.addConstr(y_in == y_out)

        # Initial ye condition
        y_output_of_in = sum(self.ye_logic[ed].item() for ed in self.node_list[0]['out'])
        self.m.addConstr(y_output_of_in == 1.0)

        # Process all possible node pairs
        for node_i in range(len(self.node_list)):
            for node_j in range(node_i + 1, len(self.node_list)):
                # Find edges between nodes i and j
                edges_between = []
                prev_node = self.node_list[node_i]
                next_node = self.node_list[node_j]
                
                # Find all edges that go from node i to node j
                for ed in prev_node['out']:
                    if ed in next_node['in']:
                        edges_between.append(ed)

                if edges_between:
                    # Build v+ and v- vectors for each edge
                    edge_vectors = []
                    
                    for ed in edges_between:          
                        # Initialize v+ and v- vectors (will only create if needed)
                        v_plus = [0] * self.dim_z
                        v_minus = [0] * self.dim_z
                        
                        # Build v+ and v- vectors based on predicates in this edge
                        for pred_tuple in self.edge_list[ed]:
                            pred_name, is_negated = pred_tuple

                            dim = self.ls_keys.index(pred_name)
                            
                            if is_negated:
                                v_minus[dim] = 1  # Set 1 in v- for negated predicate
                            else:
                                v_plus[dim] = 1   # Set 1 in v+ for non-negated predicate
                        
                        edge_vectors.append((ed, v_plus, v_minus))
                    
                    # For each dimension, enforce the constraints directly
                    for dim in range(self.dim_z):

                        # Track if any edges have v+[dim]=1 or v-[dim]=1
                        has_v_plus_edges = False
                        has_v_minus_edges = False
                        
                        # Collect edges for this dimension
                        edges_with_v_plus = []
                        edges_with_v_minus = []
                        
                        # Identify which edges have v+[dim]=1 or v-[dim]=1
                        for ed, v_plus, v_minus in edge_vectors:
                            if v_plus[dim] == 1:
                                has_v_plus_edges = True
                                edges_with_v_plus.append(ed)
                            
                            if v_minus[dim] == 1:
                                has_v_minus_edges = True
                                edges_with_v_minus.append(ed)
                        
                        # Enforce constraints only if they're non-trivial
                        if has_v_plus_edges:
                            # Create the sum expression for v+
                            sum_ye_v_plus = sum(self.ye_logic[ed].item() for ed in edges_with_v_plus)
                            self.m.addConstr(self.z_pi[dim] >= sum_ye_v_plus)

                            # # Add individual constraints for each positive edge - this is combined LNF+LT approach
                            # for ed in edges_with_v_plus:
                            #     self.m.addConstr(self.z_pi[dim] >= self.ye_logic[ed].item())
                        
                        if has_v_minus_edges:
                            # Create the sum expression for v-
                            sum_ye_v_minus = sum(self.ye_logic[ed].item() for ed in edges_with_v_minus)
                            self.m.addConstr(self.z_pi[dim] <= 1 - sum_ye_v_minus)

                            # # Add individual constraints for each negative edge
                            # for ed in edges_with_v_minus:
                            #     self.m.addConstr(self.z_pi[dim] <= 1 - self.ye_logic[ed].item())


        # Add constraints to ensure that if all predicates are satisfied, then ye=1
        for ed_idx, edge_predicates in enumerate(self.edge_list):
            if not edge_predicates:  # Skip edges with no predicates
                continue
            
            # Get the edge variable
            y_edge = self.ye_logic[ed_idx].item()
            
            # Count the total number of predicates on this edge
            num_predicates = len(edge_predicates)
            
            # Initialize sum term for the constraint
            predicate_sum = 0
            
            # Add terms for each predicate on the edge
            for pred_tuple in edge_predicates:
                pred_name, is_negated = pred_tuple
                dim = self.ls_keys.index(pred_name)
                
                if is_negated:
                    predicate_sum += (1 - self.z_pi[dim])
                else:
                    predicate_sum += self.z_pi[dim]
            
            self.m.addConstr(y_edge >= 1 - num_predicates + predicate_sum)

        self.m.update()
