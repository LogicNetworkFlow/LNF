import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import gurobipy as go
# from docplex.mp.model import Model as cplex_Model
# from docplex.mp.solution import SolveSolution

from logic_models.class_LNF import logic_nf
from logic_models.class_LT import logic_tree
from logic_models.STL.predicate import ConvexSetPredicate
from logic_models.STL.data_structure_conversion import assemble_logic_nf_info, assemble_logic_tree_info

from dynamic_models.class_DNF import dynamic_nf

from utils.temporal_graph import Node, TemporalGraph
from utils.extract_numbers import extract_numbers
from utils.polygon_map_utils import generate_random_test_cases, visualize_trajectories_on_map, get_values_from_solution, extract_trajectories

import numpy as np
import pickle
import yaml
from termcolor import colored


def main(config_path):

	# Load configuration from YAML file
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
            
	# Print configuration summary
	print("\n" + "="*80)
	print(colored(f"RUNNING CONFIGURATION: {config_path}", 'green', attrs=['bold']))
	print("="*80)
	print(f"Optimizer: {config['optimizer']['type']} - Model: {config['optimizer']['model_name']}")
	print(f"Model type: {config.get('model', 'Not specified')}")
	print(f"Graph file: {config['graph']['filename']}")
	print(f"Planning horizon: {config['horizon']}")
	print(f"Cost files: ")
	print(f"  - Hold costs: {config['costs']['hold']}")
	print(f"  - Move costs: {config['costs']['move']}")
	print(f"Initial nodes: {', '.join([f'{k}: {v}' for k, v in config['initial_nodes'].items()])}")
	print("Target nodes for YAML file:")
	print("target_nodes:")
	print(config['target_nodes'])
	print("="*80 + "\n")

	# Set up optimizer model based on configuration
	optimizer_type = config['optimizer']['type']
	if optimizer_type == "gurobi":
		optimizer_model = go.Model(config['optimizer']['model_name'])
	elif optimizer_type == "cplex":
		pass

	# Load graph data
	graph_filename = config['graph']['filename']

	# Load graph and polygon data
	with open(graph_filename, 'rb') as f:
		loaded_data = pickle.load(f)

	graph = loaded_data['graph']
	polygons = loaded_data['polygons']
	shrunk_polygons = loaded_data['shrunk_polygons']
	map_width = loaded_data['map_width']
	map_height = loaded_data['map_height']

	# Set planning horizon
	N = config['horizon']

	cost_files = config['costs']
	cost_hold = np.load(cost_files['hold'])
	cost_move = np.load(cost_files['move'])

	# Get initial nodes
	node_ini = config['initial_nodes']
	target_nodes_idx = config['target_nodes']

	dict_dnf = {}
	dnf_name = 'dnf'
	dict_dnf[dnf_name] = dynamic_nf(optimizer_model, dnf_name, N, graph, cost_move=cost_move, cost_hold=cost_hold)
	dict_dnf[dnf_name].setup_problem()
	dict_dnf[dnf_name].set_initial_condition(node_ini)
		
	# Create logic constraints
	# =====================================================================================
	# Create a list of dictionaries with com_x and com_y for each sampled node
	ls_pts = []
	for idx in target_nodes_idx:
		node = graph.nodes[idx]
		ls_pts.append({"com_x": node.config[0], "com_y": node.config[1]})

	# Print the sampled points for verification
	print("Sampled points:")
	for i, pt in enumerate(ls_pts):
		print(f"pt{i} = (x={pt['com_x']:.2f}, y={pt['com_y']:.2f})   (Node index: {target_nodes_idx[i]})")

	curr_nf_name = 'dnf'

	biped_visit0 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[0], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit2 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[2], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit4 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[4], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit6 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[6], neg=False).always(0, 3)).eventually(0, N-5)
	biped_visit8 = (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[8], neg=False).always(0, 3)).eventually(0, N-5)
	
	not_u_biped_visit0 =  (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[0], neg=True))
	u_biped_visit1 =      (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[1], neg=False))
	not_u_biped_visit2 =  (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[2], neg=True))
	u_biped_visit3 =      (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[3], neg=False))
	not_u_biped_visit4 =  (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[4], neg=True))
	u_biped_visit5 =      (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[5], neg=False))
	not_u_biped_visit6 =  (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[6], neg=True))
	u_biped_visit7 =      (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[7], neg=False))
	not_u_biped_visit8 =  (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[8], neg=True))
	u_biped_visit9 =      (ConvexSetPredicate(name=curr_nf_name, edge_no=target_nodes_idx[9], neg=False))

	spec0 = not_u_biped_visit0.until(u_biped_visit1, 0, N-5)
	spec2 = not_u_biped_visit2.until(u_biped_visit3, 0, N-5)
	spec4 = not_u_biped_visit4.until(u_biped_visit5, 0, N-5)
	spec6 = not_u_biped_visit6.until(u_biped_visit7, 0, N-5)
	spec8 = not_u_biped_visit8.until(u_biped_visit9, 0, N-5)
      
	specification = spec0 & spec2 & spec4 & spec6 & spec8 & \
		biped_visit0 & biped_visit2 & biped_visit4 & biped_visit6 & biped_visit8
	
	specification.simplify()

	# =====================================================================================
	if config['model'] == 'LNF':
            
		unique_predicate_dict, node_list, edge_list, output = assemble_logic_nf_info(specification)

		z_pi = []
		for pred_name in unique_predicate_dict.keys():
			
			nf_name, time_step, edge_no = extract_numbers(pred_name)
			z_val = dict_dnf[nf_name].ye_hold[edge_no, time_step]
			z_pi.append(z_val)
            
		_ = logic_nf(optimizer_model, unique_predicate_dict, node_list, edge_list, output, z_pi, 
			   			add_lt_cuts=False, use_sos1_encoding=False)

	elif config['model'] == 'LT':

		unique_predicate_dict, TT, UU, VV, num_extra_vars = assemble_logic_tree_info(specification)

		# Remove columns of TT that are all 0s, as they are unnecessary to create new variables.
		# TODO: We will need to fix it cleanly inside the stl.py to update STLFormula._non_predicate_counter correctly.
		non_zero_cols = []
		for j in range(TT.shape[1]):
			if not np.all(TT[:, j] == 0):
				non_zero_cols.append(j)
		TT = TT[:, non_zero_cols]
		num_extra_vars = TT.shape[1]
            
		z_pi = []
		for pred_name in unique_predicate_dict.keys():
			
			nf_name, time_step, edge_no = extract_numbers(pred_name)
			z_val = dict_dnf[nf_name].ye_hold[edge_no, time_step]
			z_pi.append(z_val)
            
		_ = logic_tree(optimizer_model, unique_predicate_dict, num_extra_vars, TT, UU, VV, z_pi)
     
	# optimizer_model.setParam('Presolve', 0)
	# optimizer_model.setParam('PreCrush', 0)
     
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
							      target_nodes_idx=target_nodes_idx, map_width=map_width,map_height=map_height, show_node_labels=False, show_time=False,
								  show_all_nodes=False, save_path=f"figures/trajectories_{os.path.splitext(os.path.basename(config_path))[0]}.png")
	
	# Memory cleanup
	optimizer_model.dispose()

	del optimizer_model, graph, polygons, shrunk_polygons, trajectories, dict_dnf

	# Force garbage collection
	import gc
	gc.collect()
	
if __name__ == '__main__':
	# No specific config provided, process all configs
	config_folder = "configs"
	
	# Find all config files
	config_files = sorted(glob.glob(f"{config_folder}/*.yaml"))
	
	total_configs = len(config_files)
	print(f"Found {total_configs} config files to process")
	
	# Process each config file
	for i, config_path in enumerate(config_files, 1):
		print(f"\nProcessing config {i}/{total_configs}: {config_path}")
		try:
			main(config_path)
		except Exception as e:
			print(f"Error processing {config_path}: {e}")
			import traceback
			traceback.print_exc()
			
	print(f"Completed processing all {total_configs} config files")
