import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.logic_models.class_LNF import logic_nf
from models.logic_models.class_LT import logic_tree
from models.logic_models.STL.predicate import ConvexSetPredicate
from models.logic_models.STL.data_structure_conversion import assemble_logic_nf_info, assemble_logic_tree_info

from models.dynamic_models.class_DNF import dynamic_nf

from utils.temporal_graph import TemporalGraph
from utils.extract_numbers import extract_numbers
from utils.polygon_map_utils import extract_trajectories
from obstacles import list_obs

import gurobipy as go
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pickle
import yaml
import time
from termcolor import colored


# =================================================================================================================================================
model = go.Model("Biped_search_and_rescue")

N = 50

# Construct bipedal network flow edges ================================
with open('traj_library_SR.pickle', 'rb') as f:
    data = pickle.load(f)

terrain_x_original, terrain_y_original, roughness_original = np.load('terrain_roughness/1_terrain_data.npz').values()
terrain_points = np.column_stack((terrain_x_original.flatten(), terrain_y_original.flatten()))
terrain_roughness = roughness_original.flatten()
tree = cKDTree(terrain_points)

ls_sol = data['ls_sol']

prim_node_list = []
prim_edge_list = []
cost_edge_list = []
lib_traj = {}

DT = 16
dt = 0.4
div_NF = DT/dt; 
for ii, sol in enumerate(data['ls_sol']):
    
    begin_point = {'com_x': round(sol['com'][0, 0].item(), 3), 'com_y': round(sol['com'][0, 1].item(), 3)}
    end_point = {'com_x': round(sol['com'][-1, 0].item(), 3), 'com_y': round(sol['com'][-1, 1].item(), 3)}
    
    if not begin_point in prim_node_list:
        prim_node_list.append(begin_point)

    if not end_point in prim_node_list:
        prim_node_list.append(end_point)

    prim_edge_list.append([prim_node_list.index(begin_point), prim_node_list.index(end_point), int(-(-(len(sol['theta'])-1)//div_NF))])
    prim_edge_list.append([prim_node_list.index(end_point), prim_node_list.index(begin_point), int(-(-(len(sol['theta'])-1)//div_NF))])

    lib_traj.update({'{}-{}'.format(prim_node_list.index(begin_point),   prim_node_list.index(end_point)) : sol})
    lib_traj.update({'{}-{}'.format(  prim_node_list.index(end_point), prim_node_list.index(begin_point)) : {key: np.flip(arr, axis=0) for key, arr in sol.items()}})

# Second pass: calculate terrain-based costs for each edge
temp_edge_costs = []
for ii, sol in enumerate(data['ls_sol']):
    # Query terrain roughness values along a robot's trajectory path, finds the nearest terrain grid point
    # distances: How far each trajectory point is from its nearest terrain point
    # indices: Index of the nearest terrain grid point for each trajectory position
    distances, indices = tree.query(sol['com'])    
    individual_costs = terrain_roughness[indices]
    
    # Convert to log-probability cost (as discussed earlier)
    failure_prob = 0.2*individual_costs / 255  # Normalize to 0-1, scale it down to 0.2 for better discrimination of the cost
    success_prob = 1 - failure_prob
    # Sum log probabilities along the trajectory
    log_success_prob = np.sum(np.log(np.maximum(success_prob, 1e-6)))  # Avoid log(0)
    
    temp_edge_costs.append(log_success_prob)
    temp_edge_costs.append(log_success_prob)  # Same cost for reverse direction
    

# Create cost matrix [N_edges x (N-1)] - time-invariant
n_edges = len(prim_edge_list)
cost_edge_matrix = np.zeros((n_edges, N-1))

for edge_idx in range(n_edges):
    edge_cost = temp_edge_costs[edge_idx]
    # Repeat the same cost across all time steps (time-invariant)
    cost_edge_matrix[edge_idx, :] = edge_cost

# If you want to use the matrix format in your dynamic_nf
cost_move = cost_edge_matrix

nb = len(prim_node_list)

def create_time_varying_severity(N=50):
   
    # Static disaster sources with individual evolution factors
    sources = [
        {"pos": (0, 20),   "intensity": 4.0, "spread": 15, "evolution": "pulsing",   "time_factor": 0.5},  # House 1 - rapid pulsing
        {"pos": (-32, 20), "intensity": 3.0, "spread": 15,  "evolution": "pulsing",  "time_factor": 0.3},  # House 2 - slow pulsing
        {"pos": (20, 20),  "intensity": 0.5, "spread": 10, "evolution": "growing",   "time_factor": 1.2},  # Airplane 1 - moderate growth
        {"pos": (40, 20),  "intensity": 0.5, "spread": 10, "evolution": "growing",   "time_factor": 2.5},  # Airplane 2 - fast growth
        {"pos": (20, 40),  "intensity": 0.5, "spread": 10, "evolution": "growing",   "time_factor": 0.8},  # Airplane 3 - slow growth
        {"pos": (-15, 40), "intensity": 0.2, "spread": 5,  "evolution": "shrinking", "time_factor": 0.6}   # Trees - moderate shrinking
    ]
    
    def severity_function(x, y, t):
        severity = 0.0
        
        for source in sources:
            # Base parameters
            base_pos = source["pos"]
            base_intensity = source["intensity"]
            base_spread = source["spread"]
            evolution = source["evolution"]
            time_factor = source["time_factor"]  # Individual evolution speed
            
            # Time-dependent modifications
            if evolution == "growing":
                # Fire spreads over time
                intensity = base_intensity * (1 + time_factor * t / N)
                spread = base_spread
                pos = base_pos
                
            elif evolution == "shrinking":
                # Fire diminishes over time
                intensity = base_intensity * max(0.1, 1 - time_factor * t / N)
                spread = base_spread
                pos = base_pos
                
            elif evolution == "pulsing":
                # Periodic danger (e.g., explosions, gas leaks)
                pulse = 0.5 * (1 + np.sin(2 * np.pi * t * time_factor / (N/3)))  # Individual pulsing speed
                intensity = base_intensity * (0.5 + 0.5 * pulse)
                spread = base_spread
                pos = base_pos
            
            # Calculate Gaussian contribution
            dx = x - pos[0]
            dy = y - pos[1]
            dist_sq = (dx**2 + dy**2) / (spread**2)
            contribution = intensity * np.exp(-0.5 * dist_sq) / (2 * np.pi * spread**2)
            severity += contribution
        
        return severity
    
    return severity_function

# Generate time-varying cost_hold
severity_func = create_time_varying_severity(N=N)


# Create cost_hold matrix [nb x (N-1)]
cost_hold = np.zeros((nb, N-1))
for node_idx, pt in enumerate(prim_node_list):
    x, y = pt['com_x'], pt['com_y']
    for t in range(N-1):
        severity_val = severity_func(x, y, t)
        # Convert to log-probability cost
        max_severity = 0.01  # Adjust based on your severity function range
        failure_prob = min(0.99, severity_val / max_severity)  # Cap at 99% failure rate
        success_prob = 1 - failure_prob
        cost_hold[node_idx, t] = np.log(success_prob)  # Negative cost (log of success probability)

# Setup for 3 robots
network_names = ['biped1', 'biped2', 'biped3']

# Create DNF for each robot with different starting positions
dict_prim_nf = {}
list_ini = [
    prim_node_list.index({'com_x': -8.0, 'com_y': 0.0}),    # Robot 1 start
    prim_node_list.index({'com_x': -32.0, 'com_y': -48.0}), # Robot 2 start  
    prim_node_list.index({'com_x': 24.0, 'com_y': -32.0})   # Robot 3 start
]

# Setup each robot
for ii, robot_name in enumerate(network_names):

    # Create a TemporalGraph object from your existing data
    graph = TemporalGraph()

    # Add nodes first using the graph's add_node method
    for i, node_data in enumerate(prim_node_list):
        config = [node_data['com_x'], node_data['com_y']]
        graph.add_node(config, name=f"node_{i}")

    # Keep track of added edges to avoid duplicates
    added_edges = set()

    # Add edges with travel times and trajectory solutions
    for edge_idx, edge_data in enumerate(prim_edge_list):
        start_idx, end_idx, travel_time = edge_data
        
        # Create edge identifier (normalized so that (A,B) and (B,A) are the same)
        edge_id = tuple(sorted([start_idx, end_idx]))
        
        # Skip if this edge pair has already been added
        if edge_id in added_edges:
            continue
        
        # Mark this edge as added
        added_edges.add(edge_id)
        
        # Get the actual node objects from the graph
        start_node = graph.nodes[start_idx]
        end_node = graph.nodes[end_idx]
        
        # Get trajectory solution
        edge_key = f'{start_idx}-{end_idx}'

        trajectory_sol = lib_traj[edge_key].copy()  # Copy to avoid modifying original
        edge_cost = cost_edge_list[edge_idx] if edge_idx < len(cost_edge_list) else 1.0
        trajectory_sol['cost'] = edge_cost

        # Add edge using node objects
        graph.add_edge(start_node, end_node, travel_time, trajectory_sol)

    node_ini = {list_ini[ii]: 1}  # Each robot starts at different location
    
    # Create dynamic_nf instance
    dict_prim_nf[robot_name] = dynamic_nf(model=model, nf_name=robot_name, N=N, graph=graph, cost_move=cost_move, cost_hold=cost_hold)
    dict_prim_nf[robot_name].setup_problem()
    dict_prim_nf[robot_name].set_initial_condition(node_ini)



# Update nf_info for multiple robots
nf_info = []
for robot_name in network_names:
   nf_info.append({robot_name + '_ye_hold': dict_prim_nf[robot_name].ye_hold})

# Define disaster sites and destinations
pt0 = {'com_x': -40.0, 'com_y': 32.0}  # House 1
pt1 = {'com_x': 16.0, 'com_y': 24.0}   # House 2
pt2 = {'com_x': 40.0, 'com_y': 16.0}   # Airplane 1
pt3 = {'com_x': 16.0, 'com_y': 40.0}   # Airplane 2
pt4 = {'com_x': -8.0, 'com_y': 40.0}   # Tree1
pt5 = {'com_x': -16.0, 'com_y': 40.0}  # Tree2
pt6 = {'com_x': -24.0, 'com_y': -24.0} # Back to camp
pt7 = {'com_x': 24.0, 'com_y': -32.0}  # Back to helicopter

n_pt0 = prim_node_list.index(pt0)
n_pt1 = prim_node_list.index(pt1)
n_pt2 = prim_node_list.index(pt2)
n_pt3 = prim_node_list.index(pt3)
n_pt4 = prim_node_list.index(pt4)
n_pt5 = prim_node_list.index(pt5)
n_pt6 = prim_node_list.index(pt6)
n_pt7 = prim_node_list.index(pt7)

# Create predicates for each robot at each location
# Robot 1 predicates
r1_pt0 = ConvexSetPredicate(name='biped1', edge_no=n_pt0, neg=False)
r1_pt1 = ConvexSetPredicate(name='biped1', edge_no=n_pt1, neg=False)
r1_pt2 = ConvexSetPredicate(name='biped1', edge_no=n_pt2, neg=False)
r1_pt3 = ConvexSetPredicate(name='biped1', edge_no=n_pt3, neg=False)
r1_pt4 = ConvexSetPredicate(name='biped1', edge_no=n_pt4, neg=False)
r1_pt5 = ConvexSetPredicate(name='biped1', edge_no=n_pt5, neg=False)
r1_pt6 = ConvexSetPredicate(name='biped1', edge_no=n_pt6, neg=False)
r1_pt7 = ConvexSetPredicate(name='biped1', edge_no=n_pt7, neg=False)

# Robot 2 predicates
r2_pt0 = ConvexSetPredicate(name='biped2', edge_no=n_pt0, neg=False)
r2_pt1 = ConvexSetPredicate(name='biped2', edge_no=n_pt1, neg=False)
r2_pt2 = ConvexSetPredicate(name='biped2', edge_no=n_pt2, neg=False)
r2_pt3 = ConvexSetPredicate(name='biped2', edge_no=n_pt3, neg=False)
r2_pt4 = ConvexSetPredicate(name='biped2', edge_no=n_pt4, neg=False)
r2_pt5 = ConvexSetPredicate(name='biped2', edge_no=n_pt5, neg=False)
r2_pt6 = ConvexSetPredicate(name='biped2', edge_no=n_pt6, neg=False)
r2_pt7 = ConvexSetPredicate(name='biped2', edge_no=n_pt7, neg=False)

# Robot 3 predicates
r3_pt0 = ConvexSetPredicate(name='biped3', edge_no=n_pt0, neg=False)
r3_pt1 = ConvexSetPredicate(name='biped3', edge_no=n_pt1, neg=False)
r3_pt2 = ConvexSetPredicate(name='biped3', edge_no=n_pt2, neg=False)
r3_pt3 = ConvexSetPredicate(name='biped3', edge_no=n_pt3, neg=False)
r3_pt4 = ConvexSetPredicate(name='biped3', edge_no=n_pt4, neg=False)
r3_pt5 = ConvexSetPredicate(name='biped3', edge_no=n_pt5, neg=False)
r3_pt6 = ConvexSetPredicate(name='biped3', edge_no=n_pt6, neg=False)
r3_pt7 = ConvexSetPredicate(name='biped3', edge_no=n_pt7, neg=False)

# Mission 1: House1 → Camp (at least one robot completes the full mission)
delta_time = 25
r1_mission1 = (r1_pt0.always(0, 4) & r1_pt6.always(delta_time, delta_time+4))
r2_mission1 = (r2_pt0.always(0, 4) & r2_pt6.always(delta_time, delta_time+4))
r3_mission1 = (r3_pt0.always(0, 4) & r3_pt6.always(delta_time, delta_time+4))
mission1 = (r1_mission1 | r2_mission1 | r3_mission1).eventually(0, N-delta_time-6)

# Mission 2: House2 → Camp (at least one robot completes the full mission)
r1_mission2 = (r1_pt1.always(0, 4) & r1_pt6.always(delta_time, delta_time+4))
r2_mission2 = (r2_pt1.always(0, 4) & r2_pt6.always(delta_time, delta_time+4))
r3_mission2 = (r3_pt1.always(0, 4) & r3_pt6.always(delta_time, delta_time+4))
mission2 = (r1_mission2 | r2_mission2 | r3_mission2).eventually(0, N-delta_time-6)

# Mission 3: Airplane1 → Helicopter (at least one robot completes the full mission)
r1_mission3 = (r1_pt2.always(0, 4) & r1_pt7.always(delta_time, delta_time+4))
r2_mission3 = (r2_pt2.always(0, 4) & r2_pt7.always(delta_time, delta_time+4))
r3_mission3 = (r3_pt2.always(0, 4) & r3_pt7.always(delta_time, delta_time+4))
mission3 = (r1_mission3 | r2_mission3 | r3_mission3).eventually(0, N-delta_time-6)

# Mission 4: Airplane2 → Helicopter (at least one robot completes the full mission)
r1_mission4 = (r1_pt3.always(0, 4) & r1_pt7.always(delta_time, delta_time+4))
r2_mission4 = (r2_pt3.always(0, 4) & r2_pt7.always(delta_time, delta_time+4))
r3_mission4 = (r3_pt3.always(0, 4) & r3_pt7.always(delta_time, delta_time+4))
mission4 = (r1_mission4 | r2_mission4 | r3_mission4).eventually(0, N-delta_time-6)

# Mission 5: Distinguish files at trees (at least one robot visits both)
r1_fire1_mission = r1_pt4.always(0, 4).eventually(0, N-6)
r2_fire1_mission = r2_pt4.always(0, 4).eventually(0, N-6)
r3_fire1_mission = r3_pt4.always(0, 4).eventually(0, N-6)
                 
r1_fire2_mission = r1_pt5.always(0, 4).eventually(0, N-6)
r2_fire2_mission = r2_pt5.always(0, 4).eventually(0, N-6)
r3_fire2_mission = r3_pt5.always(0, 4).eventually(0, N-6)

fire_mission = (r1_fire1_mission | r2_fire1_mission | r3_fire1_mission) & (r1_fire2_mission | r2_fire2_mission | r3_fire2_mission)

# r1_fire_mission = (r1_pt4.always(0, 4) & r1_pt5.always(0, 4)).eventually(0, N-6)
# r2_fire_mission = (r2_pt4.always(0, 4) & r2_pt5.always(0, 4)).eventually(0, N-6)
# r3_fire_mission = (r3_pt4.always(0, 4) & r3_pt5.always(0, 4)).eventually(0, N-6)

# fire_mission = r1_fire_mission | r2_fire_mission | r3_fire_mission

# Complete specification: all missions must be completed
specification = mission1 & mission2 & mission3 & mission4 & fire_mission
specification.simplify()

LNF_or_LT = 'LT'

if LNF_or_LT == 'LNF':

    print("Using LNF model for solving ==================================================================")

    unique_node_dict, node_list, edge_list, output = assemble_logic_nf_info(specification)

    z_pi = []
    for pred_name in unique_node_dict.keys():        
        nf_name, time_step, edge_no = extract_numbers(pred_name)
        z_val = dict_prim_nf[nf_name].ye_hold[edge_no, time_step]
        z_pi.append(z_val)

    nf_logic = logic_nf(model, unique_node_dict, node_list, edge_list, output, z_pi)

elif LNF_or_LT == 'LT':

    print("Using LT model for solving ===================================================================")

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
        z_val = dict_prim_nf[nf_name].ye_hold[edge_no, time_step]
        z_pi.append(z_val)
        
    _ = logic_tree(model, unique_predicate_dict, num_extra_vars, MT, MU, MV, z_pi)

# Solve problem ================================================================================
obj = 0.0
for name in dict_prim_nf:
    obj -= dict_prim_nf[name].obj
model.setObjective(obj, go.GRB.MINIMIZE)
model.update()

print(colored("Beginning solving ...", 'green'))
t00 = time.time()
model.optimize()
print(colored("Solving cost {} sec".format(time.time()-t00), 'green'))

if model.SolCount > 0:
    
    target_nodes_idx = [n_pt0, n_pt1, n_pt2, n_pt3, n_pt4, n_pt5, n_pt6, n_pt7]

    ls_pts = [
        {'com_x': -40.0, 'com_y': 32.0},  # pt0 - House 1
        {'com_x': 16.0, 'com_y': 24.0},   # pt1 - House 2
        {'com_x': 40.0, 'com_y': 16.0},   # pt2 - Airplane 1
        {'com_x': 16.0, 'com_y': 40.0},   # pt3 - Airplane 2
        {'com_x': -8.0, 'com_y': 40.0},   # pt4 - Tree1
        {'com_x': -16.0, 'com_y': 40.0},  # pt5 - Tree2
        {'com_x': -24.0, 'com_y': -24.0}, # pt6 - Back to camp
        {'com_x': 24.0, 'com_y': -32.0}   # pt7 - Back to helicopter
    ]

    # Extract trajectories for all robots
    all_trajectories = []
    for robot_name in network_names:
        # The "graph" is the last object created in the robot setup section
        trajectories = extract_trajectories(dnf_name=robot_name, nodes=graph.nodes, target_specs=[{"target_idx": idx} for idx in target_nodes_idx], 
            optimizer_type="gurobipy", dict_dnf=dict_prim_nf)
        all_trajectories.extend(trajectories)

    # Simple visualization
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Plot target points
    for pt in ls_pts:
        plt.scatter(pt['com_x'], pt['com_y'], color='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=10)

    # Plot obstacles
    s = 0.1
    for oo in list_obs:
        circle1 = plt.Circle((oo['pos'][0], oo['pos'][1]), oo['rad'], facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.add_patch(circle1)

    # Plot trajectories for each robot
    colors = ['blue', 'green', 'orange', 'magenta']
    for traj_idx, trajectory in enumerate(all_trajectories):
        color = colors[traj_idx % len(colors)]
        
        # Plot trajectory path
        for i in range(len(trajectory) - 1):
            current_node = trajectory[i]['vertex']
            next_node = trajectory[i+1]['vertex']
            
            if current_node != next_node:  # Only plot if moving
                # Get trajectory from library
                edge_key = f'{current_node}-{next_node}'
                sol = lib_traj[edge_key]['com']
                plt.plot(sol[:, 0], sol[:, 1], linewidth=4, color=color, alpha=0.9, zorder=5)
        
        # # Mark starting position
        # start_pos = prim_node_list[trajectory[0]['vertex']]
        # plt.scatter(start_pos['com_x'], start_pos['com_y'], s=150, color=color, marker='o', edgecolors='black', linewidth=2, zorder=8)

    # Plot terrain roughness as background
    # plt.imshow(roughness_original, 
    plt.imshow(0.2*roughness_original / 255,
            extent=[terrain_x_original.min(), terrain_x_original.max(), 
                    terrain_y_original.min(), terrain_y_original.max()],
            origin='upper',  # Fixed orientation
            cmap='terrain',
            alpha=0.7,
            zorder=0)
    plt.colorbar(label='Risk of Falling Down')

    # Create legend handles
    legend_handles = []
    for i, color in enumerate(colors[:len(network_names)]):
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=3, label=f"Robot {i+1}"))
    legend_handles.append(plt.Line2D([0], [0], marker='*', color='red', markersize=15, linestyle='None', label="Target Sites"))

    plt.legend(handles=legend_handles, loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Multi-Robot Search and Rescue Trajectories')
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.tight_layout()
    plt.savefig('path_risk_terrain.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Create two-panel plot ====================================================================================================================
    # Figure 1: Temperature heatmap at time 0 (UNCHANGED)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    severity_func = create_time_varying_severity(N=N)
    # Create a grid for heatmap
    x_grid = np.linspace(-50, 50, 100)
    y_grid = np.linspace(-50, 50, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Calculate severity at time 0 for each grid point
    max_severity = 0.01
    severity_t0 = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            severity_t0[i, j] = min(0.99, severity_func(X_grid[i, j], Y_grid[i, j], 0)/max_severity)

    # Plot heatmap
    im = ax1.imshow(severity_t0, extent=[-50, 50, -50, 50], origin='lower', cmap='YlOrRd', alpha=0.8)
    plt.colorbar(im, ax=ax1, label='Risk of Overheat', shrink=0.8)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')

    # Plot obstacles
    s = 0.1
    for oo in list_obs:
        circle1 = plt.Circle((oo['pos'][0], oo['pos'][1]), oo['rad'], facecolor='lightgray', edgecolor='black', linewidth=2)
        ax1.add_patch(circle1)

    plt.tight_layout()
    plt.savefig('risk_heatmap_t0.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Right panel: 6 separate risk curves
    fig2 = plt.figure(figsize=(12, 8))
    ax = fig2.add_subplot(111)

    # Task locations for risk evolution plots
    task_locations = [
        (-40.0, 32.0, "House 1"),
        (16.0, 24.0, "House 2"), 
        (40.0, 16.0, "Airplane 1"),
        (16.0, 40.0, "Airplane 2"),
        (-8.0, 40.0, "Trees")
    ]

    # Define colors for each curve
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    time_steps = np.arange(N)

    for idx, (x, y, name) in enumerate(task_locations):
        risk_over_time = []
        for t in time_steps:
            severity_val = severity_func(x, y, t)
            failure_prob = min(0.99, severity_val / max_severity)
            risk_over_time.append(failure_prob)
        
        ax.plot(DT*time_steps, risk_over_time, color=colors[idx], linewidth=2, label=name)

    # ax.set_title('Risk Evolution Over Time', fontsize=14)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Risk', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.4])
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('combined_task_risk_evolution.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Generate paths for XML/MuJoCo visualization
    robot_paths = {}
    traj_idx = 0
    for robot_name in network_names:
        trajectory = all_trajectories[traj_idx]  # Get trajectory for this robot
        path_coords = []
        
        for i in range(len(trajectory) - 1):
            current_vertex = trajectory[i]['vertex']
            next_vertex = trajectory[i+1]['vertex']
            
            if current_vertex != next_vertex:
                edge_key = f'{current_vertex}-{next_vertex}'
                if edge_key in lib_traj:
                    sol = lib_traj[edge_key]['com']
                    for point in sol:
                        path_coords.append([float(point[0]), float(point[1])])
        
        robot_paths[robot_name] = path_coords
        traj_idx += 1

    # Save paths to YAML file for XML generation
    import yaml
    with open('paths.yaml', 'w') as file:
        yaml.dump(robot_paths, file)

    print("Robot paths saved to paths.yaml")
    print("Now run generate_SR_paths.py to create XML files for MuJoCo visualization")
