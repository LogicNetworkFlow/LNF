import numpy as np
from typing import List, Dict, Tuple, Any
from .formula import STLFormula

def LT_to_LNF(stltree, input, node_or_edge, node_list, edge_list):
    """
    Convert Logic Tree (LT) to Logic Network Flow (LNF) representation.
    
    Args:
        stltree: The STL formula tree
        input: Current input (node or edge index)
        node_or_edge: String indicating if input is 'node' or 'edge'
        node_list: List of nodes in the network flow
        edge_list: List of edges in the network flow
        
    Returns:
        tuple: (node_or_edge, index) indicating the output type and index
    """
    
    if stltree.is_predicate():
        if node_or_edge == 'node':
            # Initiate a new edge from the node
            new_edge = [(stltree.get_name_no_inst(), stltree.neg)]
            edge_list.append(new_edge)
            node_list[input]['out'].append(len(edge_list)-1)
            return 'edge', len(edge_list)-1
        
        elif node_or_edge == 'edge':
            # Extend existing edge with this predicate
            edge_list[input].append((stltree.get_name_no_inst(), stltree.neg))
            return 'edge', input
        
        else:
            raise ValueError(f"Invalid node_or_edge value: {node_or_edge}")
    
    elif stltree.combination_type == 'and':
        # Chain edges together for AND operation
        this_input = input
        n_or_e = node_or_edge
        
        for subtree in stltree.ls_subformula:
            n_or_e, this_input = LT_to_LNF(subtree, this_input, n_or_e, node_list, edge_list)
        
        return n_or_e, this_input
    
    elif stltree.combination_type == 'or':
        # Handle OR operation by creating branches that converge at a new node
        
        if node_or_edge == 'edge':
            # Create copies of the input edge for each branch
            this_input = [input]
            for _ in range(len(stltree.ls_subformula) - 1):
                edge_list.append(edge_list[input].copy())
                this_input.append(len(edge_list) - 1)
        
        elif node_or_edge == 'node':
            # All branches start from the same node
            this_input = [input] * len(stltree.ls_subformula)
        
        else:
            raise ValueError(f"Invalid node_or_edge value: {node_or_edge}")
        
        # Process each branch
        this_output = []
        for ii, subtree in enumerate(stltree.ls_subformula):
            n_or_e, oo = LT_to_LNF(subtree, this_input[ii], node_or_edge, node_list, edge_list)
            
            # If the subtree returned a node (e.g., nested OR), create an edge from it
            if n_or_e == 'node':
                # Create an empty edge from this node
                edge_list.append([])
                node_list[oo]['out'].append(len(edge_list) - 1)
                oo = len(edge_list) - 1
            
            this_output.append(oo)
        
        # Create a convergence node where all branches meet
        node_list.append({'in': this_output, 'out': []})
        return 'node', len(node_list) - 1
    
    else:
        raise ValueError(f"Invalid combination_type: {stltree.combination_type}")


def assemble_logic_nf_info(specification):
    """
    Assemble Logic Network Flow information from an STL specification.
    
    This function converts an STL formula tree into a network flow representation
    suitable for optimization-based planning.
    
    Args:
        specification: STL formula tree (should be simplified first)
        
    Returns:
        tuple: (unique_predicate_dict, node_list, edge_list, output)
            - unique_predicate_dict: Dictionary mapping predicate names to counts
            - node_list: List of nodes in the network flow
            - edge_list: List of edges with their predicates
            - output: Index of the output node
    """
    # First ensure predicates are named
    specification.simplify()
    
    # Get unique predicate dictionary from the tree
    unique_predicate_dict = specification.unique_predicate_dict
    
    # Initialize lists for network flow
    node_list = [{'in': [], 'out': []}]
    edge_list = []
    
    # Convert tree to network flow
    __, output = LT_to_LNF(specification, 0, 'node', node_list, edge_list)
    
    return unique_predicate_dict, node_list, edge_list, output


def assemble_tree_matrices(stltree, unique_node_keys):
    """
    Assemble constraint matrices for the STL tree.
    
    This creates a matrix representation of the logical constraints that can be
    used in mixed-integer programming formulations.
    
    Args:
        stltree: The STL formula tree
        unique_node_keys: List of unique predicate keys used in the specification
        
    Returns:
        tuple: (MT, MU, MV) constraint matrices
            - MT: Tree structure constraints
            - MU: Predicate constraints
            - MV: Constant vector
    """
    
    MT = []  # Tree structure constraints
    MU = []  # Predicate constraints
    MV = []  # Constants
    
    def generate_matrices(tree):
        num_subformulas = len(tree.ls_subformula)
        
        # Initialize matrices for current node
        tt = np.zeros((num_subformulas + 1, STLFormula._non_predicate_counter))
        uu = np.zeros((num_subformulas + 1, len(unique_node_keys)))
        vv = np.zeros(num_subformulas + 1)
        
        if tree.combination_type == "and":
            # Set up main constraint
            tt[0, tree.label_non_predicate_var] = -1
            vv[0] = num_subformulas - 1  
            
            # Process each subformula
            for i, subformula in enumerate(tree.ls_subformula, 1):
                tt[i, tree.label_non_predicate_var] = 1
                
                if not subformula.is_predicate():
                    # Handle non-predicate subformula
                    tt[i, subformula.label_non_predicate_var] = -1
                    tt[0, subformula.label_non_predicate_var] = 1 
                    
                    # Recursively process subtree
                    sub_tt, sub_uu, sub_vv = generate_matrices(subformula)
                    MT.extend(sub_tt)
                    MU.extend(sub_uu)
                    MV.extend(sub_vv)
                else:
                    # Handle predicate
                    pred_name = subformula.get_name_no_inst()
                    pred_idx = unique_node_keys.index(pred_name)
                    
                    if subformula.neg:
                        # Turn z into (1-z)
                        uu[i, pred_idx] = 1
                        vv[i] = 1
                        uu[0, pred_idx] = -1
                        vv[0] -= 1
                    else:
                        uu[i, pred_idx] = -1
                        uu[0, pred_idx] = 1
                        
        else:  # OR case
            # Set up main constraint
            tt[0, tree.label_non_predicate_var] = 1
            
            # Process each subformula
            for i, subformula in enumerate(tree.ls_subformula, 1):
                tt[i, tree.label_non_predicate_var] = -1
                
                if not subformula.is_predicate():
                    # Handle non-predicate subformula
                    tt[i, subformula.label_non_predicate_var] = 1
                    tt[0, subformula.label_non_predicate_var] = -1
                    
                    # Recursively process subtree
                    sub_tt, sub_uu, sub_vv = generate_matrices(subformula)
                    MT.extend(sub_tt)
                    MU.extend(sub_uu)
                    MV.extend(sub_vv)
                else:
                    # Handle predicate
                    pred_name = subformula.get_name_no_inst()
                    pred_idx = unique_node_keys.index(pred_name)
                    
                    if subformula.neg:
                        # Turn z into (1-z)
                        uu[i, pred_idx] = -1
                        vv[i] = -1
                        uu[0, pred_idx] = 1
                        vv[0] += 1
                    else:
                        uu[i, pred_idx] = 1
                        uu[0, pred_idx] = -1
        
        return tt.tolist(), uu.tolist(), vv.tolist()
    
    # Generate initial matrices
    init_tt, init_uu, init_vv = generate_matrices(stltree)
    MT.extend(init_tt)
    MU.extend(init_uu)
    MV.extend(init_vv)
    
    # Convert lists to numpy arrays
    MT = np.array(MT)
    MU = np.array(MU)
    MV = np.array(MV).reshape(-1, 1)  # Make sure MV is 2D column vector
    
    # Add final root node constraints
    final_tt = np.zeros((2, MT.shape[1]))
    final_tt[0, -1] = 1  # The root stl variable is the last one.
    final_tt[1, -1] = -1
    
    final_uu = np.zeros((2, MU.shape[1]))
    final_vv = np.array([[1], [-1]])  # Make it a 2D column vector
    
    MT = np.vstack([MT, final_tt])
    MU = np.vstack([MU, final_uu])
    MV = np.vstack([MV, final_vv])
    
    return MT, MU, MV


def assemble_logic_tree_info(specification):
    """
    Assemble Logic Tree matrix information from an STL specification.
    
    This function converts an STL formula tree into a matrix representation
    suitable for optimization-based planning using the Logic Tree formulation.
    
    Args:
        specification: STL formula tree (should be simplified first)
        
    Returns:
        tuple: (unique_node_dict, MT, MU, MV, num_extra_vars)
            - unique_node_dict: Dictionary mapping predicate names to counts
            - MT: Tree structure constraint matrix
            - MU: Predicate constraint matrix
            - MV: Constant vector
            - num_extra_vars: Number of extra binary variables introduced
    """
    
    specification.simplify()  # This sets up predicate names
    unique_node_dict = specification.unique_predicate_dict
    unique_node_keys = list(unique_node_dict.keys())
    
    # Generate matrices
    MT, MU, MV = assemble_tree_matrices(specification, unique_node_keys)
    num_extra_vars = STLFormula._non_predicate_counter
    
    return unique_node_dict, MT, MU, MV, num_extra_vars
