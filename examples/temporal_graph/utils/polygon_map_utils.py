import numpy as np
import random
import yaml
import time

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.patches as patches
from shapely.geometry import Point
from PIL import Image
import cairosvg
import io
from termcolor import colored

def generate_random_test_cases(graph, N, num_cases=30, num_targets=10, num_robots=3):
	# Use it through: generate_random_test_cases(graph=graph, N=N, num_cases=30, num_targets=10, num_robots=3)

    # Get total number of nodes
    total_nodes = graph.get_nodes_length()
    
    # Use current time as initial seed
    seed_base = int(time.time())
    
    # Generate each test case
    for case_idx in range(1, num_cases + 1):
        # Set a new seed for each test case
        random_seed = seed_base + case_idx
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Generating test case {case_idx}/{num_cases} with seed {random_seed}")
        
        # Generate random costs
        cost_hold = np.random.uniform(0, 1, (len(graph.get_nodes_list()), N-1))
        cost_move = np.random.uniform(0, 1, (len(graph.get_edges_list()), N-1))
        
        # Save costs
        cost_hold_file = f"{case_idx}_{num_robots}robot_until_cost_hold.npy"
        cost_move_file = f"{case_idx}_{num_robots}robot_until_cost_move.npy"
        np.save(f"costs/{cost_hold_file}", cost_hold)
        np.save(f"costs/{cost_move_file}", cost_move)
        
        # Randomly sample target nodes
        target_nodes_idx = random.sample(range(total_nodes), num_targets)
        
        # Prepare config for both LNF and LT models
        config_variants = ["LNF", "LT"]
        
        for model_type in config_variants:
            # Create a config dictionary
            config = {
                "optimizer": {
                    "type": "gurobi",
                    "model_name": f"until_{num_robots}robot"
                },
                "model": model_type,
                "graph": {
                    "filename": "50_graph.pkl"
                },
                "horizon": N,
                "costs": {
                    "hold": f"costs/{cost_hold_file}",
                    "move": f"costs/{cost_move_file}"
                },
                "initial_nodes": {
                    38: 1,
                    62: 1,
                    64: 1
                },
                "target_nodes": target_nodes_idx
            }
            
            # Save the config file
            config_file = f"configs/{case_idx}_{num_robots}robot_until_{model_type.upper()}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"Created config file: {config_file}")

    print(f"Generated {num_cases*2} config files with random costs and target nodes")

def visualize_trajectories_on_map(polygons, shrunk_polygons, trajectories, nodes, target_nodes_idx, map_width, map_height, 
                                  show_node_labels=False, show_time=False, show_all_nodes=False, 
                                  save_path=None):
	"""
	Visualize robot trajectories on the polygon map.

	Args:
		polygons: Original polygon objects
		shrunk_polygons: Shrunk polygon objects (representing obstacles)
		trajectories: List of trajectories from extract_trajectories
		nodes: List of all nodes in the graph
		map_width, map_height: Dimensions of the map
		show_node_labels: Whether to show labels for nodes in trajectories
		show_time: Whether to show timestamps along trajectories
		show_all_nodes: Whether to show all nodes or just those in trajectories
		save_path: Path to save the figure (if None, just displays it)
	"""
	fig, ax = plt.subplots(figsize=(12, 12))
	
	# First, collect all polygon geometries
	polygon_geometries = []
	for polygon in shrunk_polygons:
		if hasattr(polygon, 'geoms'):  # MultiPolygon
			polygon_geometries.extend(list(polygon.geoms))
		else:  # Single Polygon
			polygon_geometries.append(polygon)

	# Create a dense grid of trees across the entire map
	tree_spacing = 2.0  # Dense spacing
	max_tree_size = 0.8  # Base max size of trees

	# For each polygon, create a dense forest strictly inside it
	for poly in polygon_geometries:
		# Draw polygon fill and outline
		x, y = poly.exterior.xy
		plt.fill(x, y, color='#e6ffe6', alpha=0.5)
		plt.plot(x, y, 'k-', linewidth=1.0)
		
		# Get polygon bounds
		minx, miny, maxx, maxy = poly.bounds
		
		# Create a very dense grid of points within the bounds
		x_grid = np.arange(minx, maxx, tree_spacing)
		y_grid = np.arange(miny, maxy, tree_spacing)
		
		# Create collections for tree parts to make rendering faster
		trunks = []
		foliage_layers = [[] for _ in range(3)]  # 3 layers of foliage for each pine tree
		
		# Tree positions within this polygon
		tree_count = 0
		
		# Create a smaller eroded polygon for checking if trees are fully inside
		# This ensures trees near the edge won't stick out
		safety_margin = 2.0  # Maximum height of a tree 
		eroded_poly = poly.buffer(-safety_margin)
		
		# If the erosion makes the polygon disappear, use the original but with smaller trees
		if eroded_poly.is_empty:
			eroded_poly = poly
		
		for x_base in x_grid:
			for y_base in y_grid:
				# Add slight jitter
				jitter_x = (np.random.random() - 0.5) * tree_spacing * 0.4
				jitter_y = (np.random.random() - 0.5) * tree_spacing * 0.4
				
				x = x_base + jitter_x
				y = y_base + jitter_y
				point = Point(x, y)
				
				# Check if point is inside the polygon
				if poly.contains(point):
					# Determine tree size based on distance to polygon edge
					in_eroded = eroded_poly.contains(point)
					
					if in_eroded:
						# Point is well inside, use full size
						size_variation = 0.8 + np.random.random() * 0.4
						size = max_tree_size * size_variation
					else:
						# Point is near edge, calculate distance to edge and scale tree accordingly
						distance_to_edge = poly.boundary.distance(point)
						# Scale size linearly with distance to edge (0 at edge, max_size at safety_margin)
						size_factor = min(1.0, distance_to_edge / safety_margin)
						size = max_tree_size * size_factor * (0.5 + np.random.random() * 0.3)
					
					# Skip if size is too small
					if size < 0.2:
						continue
					
					# Create pine tree parts
					# Trunk
					trunk_width = 0.3 * size
					trunk_height = 0.8 * size
					trunk = patches.Rectangle(
						(x - trunk_width/2, y - trunk_height/2),
						trunk_width, trunk_height,
						facecolor='saddlebrown',
						edgecolor='black',
						linewidth=0.1 * size
					)
					trunks.append(trunk)
					
					# Foliage layers - smaller triangles that won't extend much beyond the base
					for i in range(3):
						width = 1.5 * size * (1 - i*0.2)
						height = 1.2 * size
						y_pos = y + (i*height*0.3)
						triangle = patches.Polygon(
							[
								[x - width/2, y_pos],
								[x + width/2, y_pos],
								[x, y_pos + height]
							],
							closed=True,
							facecolor='darkgreen',
							edgecolor='black',
							linewidth=0.1 * size,
							alpha=0.9
						)
						foliage_layers[i].append(triangle)
					
					tree_count += 1
        
		# Add all trunks as a collection
		if trunks:
			trunk_collection = PatchCollection(trunks, match_original=True)
			ax.add_collection(trunk_collection)

		# Add foliage layers as separate collections (for proper z-ordering)
		for i, layer_patches in enumerate(foliage_layers):
			if layer_patches:
				layer_collection = PatchCollection(layer_patches, match_original=True, zorder=10+i)
				ax.add_collection(layer_collection)
            

	# Create a colormap for trajectories
	num_trajectories = len(trajectories)
	colors = cm.rainbow(np.linspace(0, 1, num_trajectories))

	# Load and convert the robot SVG once (similar to your pygame example)
	robot_image_path = '../../robot.svg'  # Path to your robot SVG
	png_data = cairosvg.svg2png(url=robot_image_path, output_width=50, output_height=50)

	# Convert to a format Matplotlib can use
	robot_img = Image.open(io.BytesIO(png_data))
	robot_array = np.array(robot_img)

	# Track which nodes are in trajectories
	nodes_in_trajectories = set()

	# Plot each trajectory
	for i, trajectory in enumerate(trajectories):
		color = colors[i]
		
		# Extract coordinates for this trajectory
		traj_coords = []
		for step in trajectory:
			vertex_idx = step['vertex']
			nodes_in_trajectories.add(vertex_idx)
			
			node = nodes[vertex_idx]
			if isinstance(node.config, np.ndarray):
				x, y = node.config[0], node.config[1]
			else:
				x, y = node.config[0], node.config[1]
			
			traj_coords.append((x, y, step['time']))

		x_init, y_init, _ = traj_coords[0]
            
		# Create a copy of the robot array to modify
		colored_robot = robot_array.copy()
		
		# Create a mask for non-transparent pixels
		mask = colored_robot[:, :, 3] > 0
		
		# Apply color - use the current trajectory color directly
		rgb = [int(c * 255) for c in color[:3]]
		for c in range(3):
			colored_robot[:, :, c][mask] = rgb[c]
		
		# Create an OffsetImage with the modified image
		imagebox = OffsetImage(colored_robot, zoom=0.5)
		
		# Create an AnnotationBbox with the OffsetImage
		ab = AnnotationBbox(imagebox, (x_init, y_init), frameon=False, pad=0.0, zorder=20)
		
		# Add to plot
		plt.gca().add_artist(ab)
		
		# Plot the trajectory
		for j in range(len(traj_coords) - 1):
			x1, y1, t1 = traj_coords[j]
			x2, y2, t2 = traj_coords[j + 1]
			
			# Draw an arrow from point j to j+1
			arrow = patches.FancyArrowPatch(
				(x1, y1), (x2, y2),
				arrowstyle='-|>', # More solid arrowhead style
				color=color,
				linewidth=3.5, # Thicker line
				mutation_scale=18, # Larger arrowhead
				shrinkA=0, # Don't shrink from start point
				shrinkB=0, # Don't shrink from end point
				zorder=15 # Higher zorder to appear on top of other elements
			)
			plt.gca().add_patch(arrow)
			
			# Show time along the trajectory if requested
			if show_time and t2 > t1 + 1:  # Only show for longer segments
				mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
				plt.text(mid_x, mid_y, f"t:{t1}→{t2}", 
							fontsize=8, color=color, 
							ha='center', va='center',
							bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

	# Plot nodes
	if show_all_nodes:
		# Plot all nodes as small dots
		for node in nodes:
			x, y = node.config[0], node.config[1]
			plt.plot(x, y, 'o', color='gray', markersize=4, alpha=0.5)

	# Plot all task nodes
	for i, node in enumerate(nodes):
		x, y = node.config[0], node.config[1]
		if i in target_nodes_idx:
			plt.plot(x, y, '*', color='yellow', markersize=25, markeredgecolor='black', markeredgewidth=2, zorder=20)

	# Plot and label nodes that are part of trajectories
	for vertex_idx in nodes_in_trajectories:
		node = nodes[vertex_idx]
		x, y = node.config[0], node.config[1]

		# if not (vertex_idx in target_nodes_idx):
		# 	plt.plot(x, y, 'o', color='black', markersize=8, zorder=10)
		
		if show_node_labels:
			if hasattr(node, 'name') and node.name:
				label = node.name
			else:
				label = str(vertex_idx)
			
			plt.text(x, y + 1, label, 
						fontsize=9, color='black', 
						ha='center', va='center', 
						bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
						zorder=11)
				
	# Plot map limit:
	map_boundary_x = [-0.5, map_width+0.5, map_width+0.5, -0.5, -0.5]
	map_boundary_y = [-0.5, -0.5, map_height+0.5, map_height+0.5, -0.5]
	plt.plot(map_boundary_x, map_boundary_y, 'k-', linewidth=2)
		
	# Add a legend for trajectories
	for i in range(num_trajectories):
		plt.plot([], [], color=colors[i], linewidth=2, label=f"Robot {i+1}")

	# Place legend outside the figure on the right
	plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Robots")

	# Adjust the figure layout to make room for the legend
	plt.tight_layout()
	plt.subplots_adjust(right=0.85)

	# Set plot limits and labels
	plt.xlim(-5, map_width+5)
	plt.ylim(-5, map_height+5)
	plt.title("Robot Trajectories on Polygon Map")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.grid(alpha=0.3)
	plt.tight_layout()

	# Save or show
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Figure saved to {save_path}")

	# plt.show()

def get_values_from_solution(optimizer_type, dnf):
	
	if optimizer_type == "gurobipy":
		sol_ys = dnf.ys.X
		sol_ye_hold = dnf.ye_hold.X
		sol_ye = [dnf.ye[i_m].X for i_m in range(len(dnf.edges))]
		
	elif optimizer_type == "cplex":
		ys_values = [var.solution_value for var in dnf.ys]
		sol_ys = np.array(ys_values)
		
		sol_ye_hold = np.zeros((dnf.nb, dnf.N_sec))
		for i in range(dnf.nb):
			for j in range(dnf.N_sec):
				sol_ye_hold[i, j] = dnf.ye_hold[i, j].solution_value
		
		sol_ye = []
		for edge_vars in dnf.ye:
			edge_values = np.array([var.solution_value for var in edge_vars])
			sol_ye.append(edge_values)
			
	return sol_ys, sol_ye_hold, sol_ye

def extract_trajectories(dnf_name, nodes, target_specs, optimizer_type, dict_dnf):
    
	# TODO: why do we need dnf name???
   
	# Get the DNF instance from the dictionary
	if dnf_name not in dict_dnf:
		raise KeyError(f"DNF instance '{dnf_name}' not found in dictionary")

	dnf = dict_dnf[dnf_name]

	# Get solution values
	sol_ys, sol_ye_hold, sol_ye = get_values_from_solution(optimizer_type, dnf)

	# Assert binary feasibility of solutions
	print(colored("Asserting all solutions are binary ...", 'green'))
	# For sol_ys
	binary_check_ys = (abs(sol_ys-1)<=1e-4) | (abs(sol_ys)<=1e-4)
	if not np.all(binary_check_ys):
		raise ValueError(f"Solution ys for network {dnf_name} is not binary. Maximum violation: {np.max(np.minimum(abs(sol_ys), abs(sol_ys-1))):.6f}")
	# For sol_ye_hold
	binary_check_ye_hold = (abs(sol_ye_hold-1)<=1e-4) | (abs(sol_ye_hold)<=1e-4)
	if not np.all(binary_check_ye_hold):
		raise ValueError(f"Solution ye_hold for network {dnf_name} is not binary. Maximum violation: {np.max(np.minimum(abs(sol_ye_hold), abs(sol_ye_hold-1))):.6f}")
	# For sol_ye
	for ss in range(len(sol_ye)):
		binary_check_ye_ss = (abs(sol_ye[ss]-1)<=1e-4) | (abs(sol_ye[ss])<=1e-4)
		if not np.all(binary_check_ye_ss):
			raise ValueError(f"Solution ye[{ss}] for network {dnf_name} is not binary. Maximum violation: {np.max(np.minimum(abs(sol_ye[ss]), abs(sol_ye[ss]-1))):.6f}")

	# Get starting vertices (where sol_ys ~= 1)
	start_vertices = np.where(sol_ys > 1-1e-4)[0]
	trajectories = []
    
	# Process each starting vertex
	for start_vertex in start_vertices:
		traj = [{'vertex': start_vertex, 'time': 0}]
		time_section = 0
		
		# Follow trajectory until end
		while time_section < dnf.N_sec:
			found = False
			
			# Check for holds
			for node_idx in range(dnf.nb):
				if (sol_ye_hold[node_idx, time_section] > 1-1e-4 and node_idx == traj[-1]['vertex']):
					traj.append({'vertex': node_idx, 'time': time_section + 1})
					time_section += 1
					found = True
					break
			
			# Check for movements
			if not found:
				for ie, ye in enumerate(sol_ye):
					if (ye.shape[0] > time_section and ye[time_section] > 1-1e-4 and dnf.edges[ie][0] == traj[-1]['vertex']):
						traj.append({'vertex': dnf.edges[ie][1], 'time': time_section + dnf.edges[ie][2]})
						time_section += dnf.edges[ie][2]
						found = True
						break
			
			if not found:
				raise Exception(f"Cannot find any edge with value one at time {time_section}!")
		
		trajectories.append(traj)
		
		# Print trajectory information
		print(f"\nTrajectory from vertex {start_vertex}")
		for state in traj:
			vertex, t = state['vertex'], state['time']
			if t == 0:
				print(f"Time 0: starting at node {vertex}, config={nodes[vertex].config[0]:.2f}, y={nodes[vertex].config[1]:.2f}")
			else:
				# Check if this is a target node
				target_indices = [spec['target_idx'] for spec in target_specs]
				if vertex in target_indices:
					target_num = target_indices.index(vertex)
					print(f"Time {t}: Node {vertex}, config={nodes[vertex].config[0]:.2f}, y={nodes[vertex].config[1]:.2f} -- reach target {target_num} !!")
				else:
					print(f"Time {t}: Node {vertex}, config={nodes[vertex].config[0]:.2f}, y={nodes[vertex].config[1]:.2f}")

	return trajectories

def plot_grid_world_with_trajectory(graph, target_groups, obstacle_nodes, initial_nodes, trajectory, grid_size=15, title="Robot Navigation", save_path=None, show_all_nodes=True):

    plt.figure(figsize=(12, 12))
    
    # Create a mapping from node index to grid coordinates
    node_to_coord = {}
    for node in graph.nodes:
        node_to_coord[node.idx] = (node.config[0], node.config[1])
    
    # Plot grid lines
    for i in range(grid_size + 1):
        plt.axhline(y=i-0.5, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)
        plt.axvline(x=i-0.5, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Plot all grid cells if requested
    if show_all_nodes:
        for node in graph.nodes:
            x, y = node.config[0], node.config[1]
            # Plot small dots for all available grid cells (non-obstacles)
            if node.idx not in obstacle_nodes:
                plt.scatter(x, y, s=20, c='lightblue', marker='o', alpha=0.3, zorder=1)
    
    # Plot obstacles as black squares
    for obs_node in obstacle_nodes:
        if obs_node in node_to_coord:
            x, y = node_to_coord[obs_node]
            plt.scatter(x, y, s=400, c='black', marker='s', 
                      edgecolors='darkgray', linewidth=1, 
                      label='Obstacles' if obs_node == obstacle_nodes[0] else "", 
                      zorder=3, alpha=0.8)
            # Add 'X' mark on obstacles
            plt.text(x, y, 'X', fontsize=12, ha='center', va='center', 
                    fontweight='bold', color='white', zorder=4)
    
    # Define colors for different target groups
    group_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot target groups with different colors and shapes
    for group_idx, group_targets in enumerate(target_groups):
        color = group_colors[group_idx % len(group_colors)]
        
        for target_node in group_targets:
            if target_node in node_to_coord:
                x, y = node_to_coord[target_node]
                plt.scatter(x, y, s=300, c=color, marker='s', 
                          edgecolors='black', linewidth=2, alpha=0.9, zorder=5,
                          label=f'Target Group {group_idx+1}' if target_node == group_targets[0] else "")
                
                # Add target labels
                plt.text(x, y-0.3, f'T{group_idx+1}', fontsize=8, ha='center', va='top', 
                        fontweight='bold', color='white', zorder=6)
    
    # Plot initial position
    for init_node in initial_nodes.keys():
        if init_node in node_to_coord:
            x, y = node_to_coord[init_node]
            plt.scatter(x, y, s=400, c='lime', marker='o', 
                      edgecolors='black', linewidth=3, 
                      label='Start', zorder=10)
            plt.text(x, y, 'S', fontsize=12, ha='center', va='center', 
                    fontweight='bold', color='black', zorder=11)
    
    # Plot trajectory
    if trajectory and len(trajectory) > 1:
        # Extract coordinates and times
        traj_x = []
        traj_y = []
        traj_times = []
        
        for node_idx, time in trajectory:
            if node_idx in node_to_coord:
                x, y = node_to_coord[node_idx]
                traj_x.append(x)
                traj_y.append(y)
                traj_times.append(time)
        
        # Plot trajectory line
        if len(traj_x) > 1:
            plt.plot(traj_x, traj_y, 'k-', linewidth=4, alpha=0.8, label='Trajectory', zorder=7)
            
            # Add arrows to show direction
            for i in range(len(traj_x) - 1):
                dx = traj_x[i+1] - traj_x[i]
                dy = traj_y[i+1] - traj_y[i]
                if dx != 0 or dy != 0:  # Only add arrow if there's movement
                    # Calculate arrow position (slightly offset from start)
                    arrow_x = traj_x[i] + dx * 0.2
                    arrow_y = traj_y[i] + dy * 0.2
                    plt.arrow(arrow_x, arrow_y, dx*0.4, dy*0.4, 
                            head_width=0.2, head_length=0.15, 
                            fc='darkred', ec='darkred', alpha=0.8, zorder=8)
            
            # Add time labels along trajectory (optional - can be cluttered)
            for i in range(0, len(traj_x), max(1, len(traj_x)//5)):  # Show every few steps
                plt.text(traj_x[i]+0.2, traj_y[i]+0.2, f't={traj_times[i]}', 
                        fontsize=8, ha='left', va='bottom', 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                        zorder=9)
        
        # Highlight final position
        if traj_x and traj_y:
            plt.scatter(traj_x[-1], traj_y[-1], s=400, c='gold', marker='*', 
                      edgecolors='black', linewidth=2, 
                      label='End', zorder=10)
            plt.text(traj_x[-1], traj_y[-1], 'E', fontsize=10, ha='center', va='center', 
                    fontweight='bold', color='black', zorder=11)
    
    # Add grid coordinate labels
    for i in range(0, grid_size, max(1, grid_size//10)):  # Don't overcrowd
        plt.text(i, -0.8, str(i), fontsize=10, ha='center', va='top')
        plt.text(-0.8, i, str(i), fontsize=10, ha='right', va='center')
    
    plt.xlim(-1.5, grid_size-0.5)
    plt.ylim(-1.5, grid_size-0.5)
    plt.gca().set_aspect('equal')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X Position', fontsize=14)
    plt.ylabel('Y Position', fontsize=14)
    
    # Add legend with better positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add comprehensive grid info text
    info_text = f"Grid: {grid_size}×{grid_size}\n"
    info_text += f"Total Nodes: {len(graph.nodes)}\n"
    info_text += f"Obstacles: {len(obstacle_nodes)}\n"
    info_text += f"Target Groups: {len(target_groups)}\n"
    info_text += f"Total Targets: {sum(len(group) for group in target_groups)}\n"
    if trajectory:
        info_text += f"Path Length: {len(trajectory)} steps\n"
        info_text += f"Total Time: {trajectory[-1][1] if trajectory else 0}"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.5))
    
    # Add a subtle grid background
    plt.gca().set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    # plt.show()
	