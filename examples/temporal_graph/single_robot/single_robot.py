import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.logic_models.class_LNF import logic_nf
from models.logic_models.class_LT import logic_tree
from models.logic_models.STL.predicate import ConvexSetPredicate
from models.logic_models.STL.data_structure_conversion import assemble_logic_nf_info, assemble_logic_tree_info

from models.dynamic_models.class_DNF import dynamic_nf
        
from utils.extract_numbers import extract_numbers
from utils.polygon_map_utils import plot_grid_world_with_trajectory

import gurobipy as go
import numpy as np
import pickle
import random


def generate_random_scenario(graph, Ng, Nt=2, No_factor=2, rand_seed=None):
   
    if rand_seed is not None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
    
    total_nodes = graph.get_nodes_length()
    No = No_factor * Ng
    
    # Sample obstacle nodes
    obstacle_nodes = random.sample(range(total_nodes), min(No, total_nodes // 4))
    
    # Remaining nodes for targets and initial positions
    available_nodes = [i for i in range(total_nodes) if i not in obstacle_nodes]
    
    # Sample target groups
    target_groups = []
    used_targets = set()
    
    for group_idx in range(Ng):
        # Sample Nt targets for this group
        available_for_group = [n for n in available_nodes if n not in used_targets]
        if len(available_for_group) < Nt:
            # If not enough nodes, use what's available
            group_targets = available_for_group
        else:
            group_targets = random.sample(available_for_group, Nt)
        
        target_groups.append(group_targets)
        used_targets.update(group_targets)
    
    # Sample initial robot position (single robot)
    available_for_init = [n for n in available_nodes if n not in used_targets]
    init_position = random.sample(available_for_init, 1)[0]
    
    initial_nodes = {init_position: 1}  # Single robot at one position
    
    return target_groups, obstacle_nodes, initial_nodes

def run_single_scenario(graph, Ng, horizon=15, trial_idx=0, grid_size=16, save_path=None):
    
    print(f"\n{'='*70}")
    print(f"SCENARIO: Ng={Ng}, Trial={trial_idx} - Running BOTH LNF and LT")
    print(f"{'='*70}")
    
    # Generate random scenario (same for both models)
    rand_seed = 42 + trial_idx
    target_groups, obstacle_nodes, initial_nodes = generate_random_scenario(graph, Ng, Nt=3, No_factor=2, rand_seed=rand_seed)
    
    # Generate random costs (same for both models)
    np.random.seed(rand_seed)
    cost_hold = np.random.uniform(0, 1, (graph.get_nodes_length(), horizon-1))
    cost_move = np.random.uniform(0, 1, (graph.get_edges_length(), horizon-1))
    # cost_hold = np.ones((graph.get_nodes_length(), horizon-1))*0
    # cost_move = np.ones((graph.get_edges_length(), horizon-1))

    print(f"Generated scenario:")
    print(f"  - Target groups: {Ng}")
    print(f"  - Total targets: {sum(len(group) for group in target_groups)}")
    print(f"  - Obstacles: {len(obstacle_nodes)}")
    print(f"  - Initial robot position: {list(initial_nodes.keys())[0]}")
    
    results = []
    
    # Run both models on the same scenario
    for model_type in ['LNF', 'LT']:
        print(f"\n{'-'*50}")
        print(f"Running {model_type} model...")
        print(f"{'-'*50}")
        
        try:
            # Set up optimization model
            optimizer_model = go.Model(f"benchmark_Ng{Ng}_{model_type}_trial{trial_idx}")
            
            # Create dynamic network flow
            dict_dnf = {}
            dnf_name = 'dnf'
            dict_dnf[dnf_name] = dynamic_nf(optimizer_model, dnf_name, horizon, graph, cost_move=cost_move, cost_hold=cost_hold)
            dict_dnf[dnf_name].setup_problem()
            dict_dnf[dnf_name].set_initial_condition(initial_nodes)
            
            # Part 1: Always avoid obstacles - □[0,T]¬obstacle
            obstacle_avoidance_specs = []
            for obs_node in obstacle_nodes:
                # Never visit obstacle nodes
                obs_pred = ConvexSetPredicate(name=dnf_name, edge_no=obs_node, neg=True)
                obstacle_avoidance = obs_pred.always(0, horizon-2)
                obstacle_avoidance_specs.append(obstacle_avoidance)

            # Part 2: Visit target groups
            group_specs = []
            for group_idx, group_targets in enumerate(target_groups):
                target_predicates = []
                for target_node in group_targets:
                    target_pred = ConvexSetPredicate(name=dnf_name, edge_no=target_node, neg=False).always(0, 1)
                    target_predicates.append(target_pred)
                
                # Disjunction of targets in this group
                group_disjunction = target_predicates[0]
                for pred in target_predicates[1:]:
                    group_disjunction = group_disjunction | pred
                
                # Eventually visit one of the targets in this group
                group_eventually = group_disjunction.eventually(0, horizon-5)
                group_specs.append(group_eventually)

            # Combine obstacle avoidance with target visiting
            all_specs = obstacle_avoidance_specs + group_specs

            # Create the full specification (conjunction of all constraints)
            specification = all_specs[0]
            for spec in all_specs[1:]:
                specification = specification & spec

            specification.simplify()
            
            # Set up logic constraints based on model type
            if model_type == 'LNF':
                unique_predicate_dict, node_list, edge_list, output = assemble_logic_nf_info(specification)
                
                z_pi = []
                for pred_name in unique_predicate_dict.keys():
                    nf_name_pred, time_step, edge_no = extract_numbers(pred_name)
                    z_val = dict_dnf[nf_name_pred].ye_hold[edge_no, time_step]
                    z_pi.append(z_val)
                
                _ = logic_nf(optimizer_model, unique_predicate_dict, node_list, edge_list, output, z_pi)
                
            elif model_type == 'LT':
                unique_predicate_dict, TT, UU, VV, num_extra_vars = assemble_logic_tree_info(specification)
                
                # Remove columns of TT that are all 0s
                non_zero_cols = []
                for j in range(TT.shape[1]):
                    if not np.all(TT[:, j] == 0):
                        non_zero_cols.append(j)
                TT = TT[:, non_zero_cols]
                num_extra_vars = TT.shape[1]
                
                z_pi = []
                for pred_name in unique_predicate_dict.keys():
                    nf_name_pred, time_step, edge_no = extract_numbers(pred_name)
                    z_val = dict_dnf[nf_name_pred].ye_hold[edge_no, time_step]
                    z_pi.append(z_val)
                
                _ = logic_tree(optimizer_model, unique_predicate_dict, num_extra_vars, TT, UU, VV, z_pi)
            
            # Set objective and solve
            obj = dict_dnf[dnf_name].obj
            optimizer_model.setObjective(obj, go.GRB.MINIMIZE)
            optimizer_model.update()
            
            print(f"{model_type} model setup complete. Starting optimization...")
            
            optimizer_model.optimize()

            if optimizer_model.solCount > 0:
                # Extract trajectory
                trajectory = extract_trajectory_from_solution(dict_dnf[dnf_name], graph, horizon)
                
                # Plot the result
                plot_title = f"{model_type} Solution: Ng={Ng}, Trial={trial_idx}"
                save_name = f"figures/trajectory_Ng{Ng}_trial{trial_idx}_{model_type}.png"
                
                plot_grid_world_with_trajectory(graph=graph, target_groups=target_groups, obstacle_nodes=obstacle_nodes, initial_nodes=initial_nodes, 
                                                trajectory=trajectory, grid_size=grid_size, title=plot_title, save_path=save_name)
            
        except Exception as e:
            print(f"Error in {model_type} experiment: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison for this scenario
    print(f"\n{'='*50}")
    print(f"SCENARIO COMPARISON (Ng={Ng}, Trial={trial_idx}):")
    print(f"{'='*50}")
    if len(results) == 2 and results[0]['status'] != 'ERROR' and results[1]['status'] != 'ERROR':
        lnf_result = results[0] if results[0]['model_type'] == 'LNF' else results[1]
        lt_result = results[1] if results[1]['model_type'] == 'LT' else results[0]
        
        print(f"LNF: {lnf_result['solve_time']:.2f}s, {lnf_result['binary_vars']} binary vars")
        print(f"LT:  {lt_result['solve_time']:.2f}s, {lt_result['binary_vars']} binary vars")
        
        if lnf_result['solve_time'] and lt_result['solve_time']:
            speedup = lnf_result['solve_time'] / lt_result['solve_time']
            faster_model = "LT" if speedup > 1 else "LNF"
            print(f"Speedup: {abs(speedup):.2f}x faster with {faster_model}")
    
    return results

def extract_trajectory_from_solution(dnf, graph, horizon):
   
    try:
        # Get solution values
        sol_ys = dnf.ys.X
        sol_ye_hold = dnf.ye_hold.X
        sol_ye = [dnf.ye[i_m].X for i_m in range(len(dnf.edges))]
        
        # Find starting position
        start_nodes = np.where(sol_ys > 0.5)[0]
        if len(start_nodes) == 0:
            return []
        
        start_node = start_nodes[0]
        trajectory = [(start_node, 0)]
        
        current_node = start_node
        current_time = 0
        
        # Follow the trajectory
        while current_time < horizon - 1:
            found_move = False
            
            # Check for holds first
            if current_time < sol_ye_hold.shape[1] and sol_ye_hold[current_node, current_time] > 0.5:
                current_time += 1
                trajectory.append((current_node, current_time))
                found_move = True
            else:
                # Check for movements
                for edge_idx, ye_vals in enumerate(sol_ye):
                    if (current_time < len(ye_vals) and 
                        ye_vals[current_time] > 0.5 and 
                        dnf.edges[edge_idx][0] == current_node):
                        
                        # Move to next node
                        next_node = dnf.edges[edge_idx][1]
                        travel_time = dnf.edges[edge_idx][2]
                        current_time += travel_time
                        current_node = next_node
                        
                        if current_time < horizon:
                            trajectory.append((current_node, current_time))
                        found_move = True
                        break
            
            if not found_move:
                break
        
        return trajectory
        
    except Exception as e:
        print(f"Error extracting trajectory: {e}")
        return []

if __name__ == "__main__":
    # Run the benchmark experiment
    graph_file='grid_world_16x16.pkl'
    Ng_values=[2, 3, 4, 5]  # Test specification complexity
    num_trials=10  # Number of trials per Ng
    
    print("="*80)
    print("STARTING BENCHMARK EXPERIMENT - LNF vs LT COMPARISON")
    print("="*80)
    print(f"Graph file: {graph_file}")
    print(f"Ng values: {Ng_values}")
    print(f"Trials per Ng: {num_trials}")
    print(f"Models: LNF and LT (both run on each scenario)")
    print("="*80)
    
    # Load graph
    if graph_file.endswith('.pkl'):
        with open(graph_file, 'rb') as f:
            data = pickle.load(f)
            graph = data['graph']
    else:
        raise ValueError("Unsupported graph file format")
    
    print(f"Loaded graph: {graph.get_nodes_length()} nodes, {graph.get_edges_length()} edges")
    
    # Run experiments
    total_scenarios = len(Ng_values) * num_trials
    scenario_count = 0

    for Ng in Ng_values:
        for trial in range(num_trials):
            scenario_count += 1
            save_path = f"loggings/scenario_Ng{Ng}_trial{trial}"
            print(f"\nRunning scenario {scenario_count}/{total_scenarios}")
            horizon=Ng*10
            run_single_scenario(graph=graph, Ng=Ng, horizon=horizon, trial_idx=trial, grid_size=16, save_path=save_path)
