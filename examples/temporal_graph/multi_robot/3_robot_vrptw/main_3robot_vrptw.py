import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from models.logic_models.class_LNF import logic_nf
from models.logic_models.class_LT import logic_tree
from models.logic_models.STL.predicate import ConvexSetPredicate
from models.logic_models.STL.data_structure_conversion import assemble_logic_nf_info, assemble_logic_tree_info

from models.dynamic_models.class_DNF import dynamic_nf

from utils.extract_numbers import extract_numbers
from utils.polygon_map_utils import visualize_trajectories_on_map, extract_trajectories

import gurobipy as go
import numpy as np
import pickle
import random
from termcolor import colored


def main():

	script_dir = os.path.dirname(os.path.abspath(__file__))
	graph_filename = os.path.join(script_dir, "..", "50_graph.pkl")
	N = 50
	num_robots = 3
	num_targets = 10

	# Load graph and polygon data
	with open(graph_filename, 'rb') as f:
		loaded_data = pickle.load(f)

	graph = loaded_data['graph']
	polygons = loaded_data['polygons']
	shrunk_polygons = loaded_data['shrunk_polygons']
	map_width = loaded_data['map_width']
	map_height = loaded_data['map_height']

	num_nodes = graph.get_nodes_length()
	num_edges = graph.get_edges_length()

	# Generate random costs
	cost_hold = np.random.uniform(0, 1, (num_nodes, N-1))
	cost_move = np.random.uniform(0, 1, (num_edges, N-1))

	# Generate random initial nodes
	node_ini = {idx: 1 for idx in random.sample(range(num_nodes), num_robots)}

	# Generate random target nodes
	target_nodes_idx = random.sample(range(num_nodes), num_targets)

	curr_nf_name = 'dnf'

	biped_visit0 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[0], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit1 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[1], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit2 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[2], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit3 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[3], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit4 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[4], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit5 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[5], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit6 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[6], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit7 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[7], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit8 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[8], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit9 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[9], neg=False).always(0, 3)).eventually(0, N-5)
      
	specification = biped_visit0 & biped_visit1 & biped_visit2 & biped_visit3 & biped_visit4 & \
		biped_visit5 & biped_visit6 & biped_visit7 & biped_visit8 & biped_visit9
	
	specification.simplify()

	# =====================================================================================
	for model_type in ['LNF', 'LT']:
		print(f"\n{'='*80}")
		print(colored(f"Running {model_type} model", 'green', attrs=['bold']))
		print(f"{'='*80}\n")

		# Create optimizer model
		optimizer_model = go.Model(f"vrptw_{num_robots}robot_{model_type}")

		dict_dnf = {}
		dnf_name = 'dnf'
		dict_dnf[dnf_name] = dynamic_nf(optimizer_model, dnf_name, N, graph, cost_move=cost_move, cost_hold=cost_hold)
		dict_dnf[dnf_name].setup_problem()
		dict_dnf[dnf_name].set_initial_condition(node_ini)
		
		if model_type == 'LNF':
				
			unique_predicate_dict, node_list, edge_list, output = assemble_logic_nf_info(specification)

			z_pi = []
			for pred_name in unique_predicate_dict.keys():
				
				nf_name, time_step, edge_no = extract_numbers(pred_name)
				z_val = dict_dnf[nf_name].ye_hold[edge_no, time_step]
				z_pi.append(z_val)
				
			_ = logic_nf(optimizer_model, unique_predicate_dict, node_list, edge_list, output, z_pi, 
							add_lt_cuts=False, use_sos1_encoding=False)

		elif model_type == 'LT':

			unique_predicate_dict, MT, MU, MV, num_extra_vars = assemble_logic_tree_info(specification)

			# Remove columns of MT that are all 0s, as they are unnecessary to create new variables.
			# TODO: We will need to fix it cleanly inside the stl.py to update STLFormula._non_predicate_counter correctly.
			non_zero_cols = []
			for j in range(MT.shape[1]):
				if not np.all(MT[:, j] == 0):
					non_zero_cols.append(j)
			MT = MT[:, non_zero_cols]
			num_extra_vars = MT.shape[1]
				
			z_pi = []
			for pred_name in unique_predicate_dict.keys():
				
				nf_name, time_step, edge_no = extract_numbers(pred_name)
				z_val = dict_dnf[nf_name].ye_hold[edge_no, time_step]
				z_pi.append(z_val)
				
			_ = logic_tree(optimizer_model, unique_predicate_dict, num_extra_vars, MT, MU, MV, z_pi)
     
		obj = dict_dnf[nf_name].obj

		optimizer_model.setObjective(obj, go.GRB.MINIMIZE)
		optimizer_model.update()

		# Solve the model
		print("Solving the problem with temporal logic constraints...")
		optimizer_model.optimize()
		
		# solution = solver(optimizer_model, dict_dynamic_nf, current_experiment, log_idx)
		trajectories = extract_trajectories(dnf_name=dnf_name, nodes=graph.nodes, target_specs=[{"target_idx": idx} for idx in target_nodes_idx], 
											optimizer_type="gurobipy", dict_dnf=dict_dnf)
		
		visualize_trajectories_on_map(polygons=polygons, shrunk_polygons=shrunk_polygons, trajectories=trajectories, nodes=graph.nodes, 
									target_nodes_idx=target_nodes_idx, map_width=map_width, map_height=map_height, show_node_labels=False, show_time=False,
									show_all_nodes=False, save_path=f"trajectories_VRPTW_{model_type}.png")
	
if __name__ == '__main__':
	main()