import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from docplex.mp.model import Model as cplex_Model
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

        # Convert z_pi variables to binary for CPLEX
        for zz in self.z_pi:
            zz.set_vartype('B')  # Set as binary in DOcplex
        
        # Create ye_logic variables for CPLEX
        self.ye_logic = []
        for i in range(self.num_edges):
            var = model.continuous_var(lb=0.0, ub=1.0, name=f'ye_logic_{i}')
            self.ye_logic.append(var)
        
        print(colored('ye_logic are not explicitly enforced as binary variables!', 'red'))
        print("Setting up combined flow structure with CPLEX...")
        self.enforce_flow_structure()

    def enforce_flow_structure(self):
        # Enforce shared ye_logic flow conservation
        for nn in self.node_list:
            if nn['in'] != [] and nn['out'] != []:
                y_in_terms = [self.ye_logic[inn] for inn in nn['in']]
                y_out_terms = [self.ye_logic[ouu] for ouu in nn['out']]
                
                y_in = self.m.sum(y_in_terms) if len(y_in_terms) > 1 else y_in_terms[0]
                y_out = self.m.sum(y_out_terms) if len(y_out_terms) > 1 else y_out_terms[0]
                
                self.m.add_constraint(y_in == y_out, ctname="flow_conservation")

        # Initial ye condition
        initial_terms = [self.ye_logic[ed] for ed in self.node_list[0]['out']]
        y_output_of_in = self.m.sum(initial_terms) if len(initial_terms) > 1 else initial_terms[0]
        self.m.add_constraint(y_output_of_in == 1.0, ctname="initial_condition")

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
                        # Initialize v+ and v- vectors
                        v_plus = [0] * self.dim_z
                        v_minus = [0] * self.dim_z
                        
                        # Build v+ and v- vectors based on predicates in this edge
                        for pred_tuple in self.edge_list[ed]:
                            pred_name, is_negated = pred_tuple
                            dim = self.ls_keys.index(pred_name)
                            
                            if is_negated:
                                v_minus[dim] = 1
                            else:
                                v_plus[dim] = 1
                        
                        edge_vectors.append((ed, v_plus, v_minus))
                    
                    # For each dimension, enforce the constraints
                    for dim in range(self.dim_z):
                        # Collect edges for this dimension
                        edges_with_v_plus = [ed for ed, v_plus, _ in edge_vectors if v_plus[dim] == 1]
                        edges_with_v_minus = [ed for ed, _, v_minus in edge_vectors if v_minus[dim] == 1]
                        
                        # Enforce constraints only if they're non-trivial
                        if edges_with_v_plus:
                            v_plus_terms = [self.ye_logic[ed] for ed in edges_with_v_plus]
                            sum_ye_v_plus = self.m.sum(v_plus_terms) if len(v_plus_terms) > 1 else v_plus_terms[0]
                            self.m.add_constraint(self.z_pi[dim] >= sum_ye_v_plus, ctname=f"v_plus_constraint_{node_i}_{node_j}_{dim}")
                        
                        if edges_with_v_minus:
                            v_minus_terms = [self.ye_logic[ed] for ed in edges_with_v_minus]
                            sum_ye_v_minus = self.m.sum(v_minus_terms) if len(v_minus_terms) > 1 else v_minus_terms[0]
                            self.m.add_constraint(self.z_pi[dim] <= 1 - sum_ye_v_minus, ctname=f"v_minus_constraint_{node_i}_{node_j}_{dim}")

        # Add constraints to ensure that if all predicates are satisfied, then ye=1
        for ed_idx, edge_predicates in enumerate(self.edge_list):
            if not edge_predicates:  # Skip edges with no predicates
                continue
            
            y_edge = self.ye_logic[ed_idx]
            num_predicates = len(edge_predicates)
            
            # Build predicate sum
            predicate_terms = []
            for pred_tuple in edge_predicates:
                pred_name, is_negated = pred_tuple
                dim = self.ls_keys.index(pred_name)
                
                if is_negated:
                    predicate_terms.append(1 - self.z_pi[dim])
                else:
                    predicate_terms.append(self.z_pi[dim])
            
            if predicate_terms:
                predicate_sum = self.m.sum(predicate_terms) if len(predicate_terms) > 1 else predicate_terms[0]
                self.m.add_constraint(y_edge >= 1 - num_predicates + predicate_sum, ctname=f"predicate_satisfaction_{ed_idx}")

        # Additional inequality constraints from flow conservation
        for node_idx in range(1, len(self.node_list) - 1):  # Skip source and sink nodes
            node = self.node_list[node_idx]
            in_edges = node['in']
            out_edges = node['out']
            
            # Skip nodes with no in or out edges
            if not in_edges or not out_edges:
                continue
            
            # Construct v+ and v- for each input edge
            in_edge_vectors = []
            for ed in in_edges:
                v_plus = [0] * self.dim_z
                v_minus = [0] * self.dim_z
                
                for pred_tuple in self.edge_list[ed]:
                    pred_name, is_negated = pred_tuple
                    dim = self.ls_keys.index(pred_name)
                    
                    if is_negated:
                        v_minus[dim] = 1
                    else:
                        v_plus[dim] = 1
                
                in_edge_vectors.append((ed, v_plus, v_minus))
            
            # Construct v+ and v- for each output edge
            out_edge_vectors = []
            for ed in out_edges:
                v_plus = [0] * self.dim_z
                v_minus = [0] * self.dim_z
                
                for pred_tuple in self.edge_list[ed]:
                    pred_name, is_negated = pred_tuple
                    dim = self.ls_keys.index(pred_name)
                    
                    if is_negated:
                        v_minus[dim] = 1
                    else:
                        v_plus[dim] = 1
                
                out_edge_vectors.append((ed, v_plus, v_minus))
            
            # For each dimension, add flow conservation constraints
            for dim in range(self.dim_z):
                # Count edges with v+ and v- for this dimension
                in_edges_with_v_plus = [ed for ed, v_plus, _ in in_edge_vectors if v_plus[dim] == 1]
                in_edges_with_v_minus = [ed for ed, _, v_minus in in_edge_vectors if v_minus[dim] == 1]
                out_edges_with_v_plus = [ed for ed, v_plus, _ in out_edge_vectors if v_plus[dim] == 1]
                out_edges_with_v_minus = [ed for ed, _, v_minus in out_edge_vectors if v_minus[dim] == 1]
                
                # Case 1: input has v+, output has v-
                if in_edges_with_v_plus and out_edges_with_v_minus:
                    in_v_plus_terms = [self.ye_logic[ed] for ed in in_edges_with_v_plus]
                    out_v_minus_terms = [self.ye_logic[ed] for ed in out_edges_with_v_minus]
                    
                    sum_in_v_plus = self.m.sum(in_v_plus_terms) if len(in_v_plus_terms) > 1 else in_v_plus_terms[0]
                    sum_out_v_minus = self.m.sum(out_v_minus_terms) if len(out_v_minus_terms) > 1 else out_v_minus_terms[0]
                    
                    if len(in_edges_with_v_plus) > 0 and len(out_edges_with_v_minus) > 0:
                        self.m.add_constraint(sum_in_v_plus <= 1 - sum_out_v_minus, ctname=f"flow_constraint_1_{node_idx}_{dim}")
                
                # Case 2: input has v-, output has v+
                if in_edges_with_v_minus and out_edges_with_v_plus:
                    in_v_minus_terms = [self.ye_logic[ed] for ed in in_edges_with_v_minus]
                    out_v_plus_terms = [self.ye_logic[ed] for ed in out_edges_with_v_plus]
                    
                    sum_in_v_minus = self.m.sum(in_v_minus_terms) if len(in_v_minus_terms) > 1 else in_v_minus_terms[0]
                    sum_out_v_plus = self.m.sum(out_v_plus_terms) if len(out_v_plus_terms) > 1 else out_v_plus_terms[0]
                    
                    if len(in_edges_with_v_minus) > 0 and len(out_edges_with_v_plus) > 0:
                        self.m.add_constraint(1 - sum_in_v_minus >= sum_out_v_plus, ctname=f"flow_constraint_2_{node_idx}_{dim}")

