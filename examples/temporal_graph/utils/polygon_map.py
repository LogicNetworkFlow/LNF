import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, distance
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union, linemerge
from utils.temporal_graph import TemporalGraph
import math
import random
import pickle

def generate_weighted_voronoi_polygons(width, height, num_points=30, rand_seed=None):
    """
    Generate irregular polygons using weighted Voronoi tessellation.
    
    Parameters:
    - width, height: Dimensions of the map
    - num_points: Number of seed points
    - rand_seed: Random seed for reproducibility
    
    Returns:
    - List of shapely.Polygon objects
    """
    if rand_seed is not None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
    
    # Generate random points inside the area with an extra buffer
    buffer = max(width, height) * 0.1
    extended_width = width + 2 * buffer
    extended_height = height + 2 * buffer
    
    points = []
    for _ in range(num_points):
        x = random.uniform(-buffer, width + buffer)
        y = random.uniform(-buffer, height + buffer)
        points.append((x, y))
    
    # Create a boundary polygon
    boundary = Polygon([
        (0, 0), (width, 0), (width, height), (0, height)
    ])
    
    # Create Voronoi diagram
    vor = Voronoi(np.array(points))
    
    # Extract Voronoi polygons
    voronoi_polygons = []
    
    for region_idx, region in enumerate(vor.regions):
        if not region or -1 in region:  # Skip empty regions or regions extending to infinity
            continue
        
        # Get vertices of the region
        vertices = [vor.vertices[i] for i in region]
        if len(vertices) < 3:  # Skip degenerate polygons
            continue
        
        # Create polygon from vertices
        polygon = Polygon(vertices)
        
        # Only keep the part of the polygon that's inside the map boundary
        clipped_polygon = polygon.intersection(boundary)
        
        if not clipped_polygon.is_empty and clipped_polygon.area > 0:
            voronoi_polygons.append(clipped_polygon)
    
    return voronoi_polygons

def shrink_polygons(polygons, shrink_distance=2.0):
    """
    Shrink each polygon to create paths between them.
    
    Parameters:
    - polygons: List of shapely.Polygon objects
    - shrink_distance: Distance to shrink each polygon
    
    Returns:
    - List of shrunk polygons
    """
    shrunk_polygons = []
    for polygon in polygons:
        # Handle both Polygon and MultiPolygon
        if isinstance(polygon, Polygon):
            shrunk = polygon.buffer(-shrink_distance)
            if not shrunk.is_empty:
                shrunk_polygons.append(shrunk)
        elif isinstance(polygon, MultiPolygon):
            for p in polygon.geoms:
                shrunk = p.buffer(-shrink_distance)
                if not shrunk.is_empty:
                    shrunk_polygons.append(shrunk)
    
    return shrunk_polygons

# Create a more detailed visualization function
def visualize_map_detailed(original_polygons, shrunk_polygons, map_width, map_height):
    """
    Visualize the map with original polygons, shrunk polygons, and paths with more detail.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Original polygons with their divisions
    ax = axes[0]
    
    for polygon in original_polygons:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            ax.fill(x, y, color=np.random.rand(3,), alpha=0.3)  # Random color for each polygon
            ax.plot(x, y, 'k-', linewidth=1.5)
        elif isinstance(polygon, MultiPolygon):
            for p in polygon.geoms:
                x, y = p.exterior.xy
                ax.fill(x, y, color=np.random.rand(3,), alpha=0.3)
                ax.plot(x, y, 'k-', linewidth=1.5)
    
    ax.set_title("Original Map Division")
    ax.set_aspect('equal')
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.axis('off')
    
    # Plot 2: Shrunk polygons with paths
    ax = axes[1]
    
    # # First plot original polygon boundaries as paths
    # for polygon in original_polygons:
    #     if isinstance(polygon, Polygon):
    #         x, y = polygon.exterior.xy
    #         ax.plot(x, y, 'k-', linewidth=1.5)
    #     elif isinstance(polygon, MultiPolygon):
    #         for p in polygon.geoms:
    #             x, y = p.exterior.xy
    #             ax.plot(x, y, 'k-', linewidth=1.5)
    
    # Then plot shrunk polygons as obstacles
    for polygon in shrunk_polygons:
        if isinstance(polygon, Polygon):
            x, y = polygon.exterior.xy
            ax.fill(x, y, color='lightgray', alpha=0.8)
            ax.plot(x, y, 'k-', linewidth=0.8)
        elif isinstance(polygon, MultiPolygon):
            for p in polygon.geoms:
                x, y = p.exterior.xy
                ax.fill(x, y, color='lightgray', alpha=0.8)
                ax.plot(x, y, 'k-', linewidth=0.8)
    
    ax.set_title("Map with Shrunk Polygons and Paths")
    ax.set_aspect('equal')
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def convert_polygons_to_temporal_graph(polygons, velocity=1.0, min_distance=0.01):
    """
    Convert a polygon map to a TemporalGraph with ordered vertex numbering.
    Vertices are numbered by polygon position, from top-left to bottom-right.
    
    Parameters:
    - polygons: List of shapely.Polygon objects
    - velocity: Constant velocity for travel time calculation
    - min_distance: Minimum distance between nodes to consider them distinct
    
    Returns:
    - TemporalGraph object
    """
    # Create a new temporal graph
    graph = TemporalGraph()
    
    # Step 1: Calculate polygon centroids and sort them top-left to bottom-right
    polygon_centroids = []
    for i, polygon in enumerate(polygons):
        centroid = polygon.centroid if isinstance(polygon, Polygon) else list(polygon.geoms)[0].centroid
        polygon_centroids.append((i, centroid.x, centroid.y))
    
    # Sort polygons by row then column (top-left to bottom-right)
    map_height = max(y for _, _, y in polygon_centroids)
    cols = int(np.sqrt(len(polygon_centroids)))
    sorted_polygons = sorted(polygon_centroids, key=lambda p: (int(p[2] / (map_height / cols)), p[1]))
    
    # Step 2: Extract all boundary lines and vertices, preserving polygon order
    all_lines = []
    vertex_id_map = {}  # Maps (x, y) coordinates to vertex IDs
    vertex_id = 0  # Start from 0 to match node indices
    
    for poly_idx, _, _ in sorted_polygons:
        polygon = polygons[poly_idx]
        
        if isinstance(polygon, Polygon):
            coords = list(polygon.exterior.coords)
            
            # Add vertex IDs
            for i, coord in enumerate(coords[:-1]):  # Skip last point (same as first)
                rounded = (round(coord[0], 10), round(coord[1], 10))
                if rounded not in vertex_id_map:
                    vertex_id_map[rounded] = vertex_id
                    vertex_id += 1
            
            # Add lines
            for i in range(len(coords) - 1):
                line = LineString([coords[i], coords[i+1]])
                all_lines.append(line)
                
        elif isinstance(polygon, MultiPolygon):
            for p in polygon.geoms:
                coords = list(p.exterior.coords)
                
                # Add vertex IDs
                for i, coord in enumerate(coords[:-1]):
                    rounded = (round(coord[0], 10), round(coord[1], 10))
                    if rounded not in vertex_id_map:
                        vertex_id_map[rounded] = vertex_id
                        vertex_id += 1
                
                # Add lines
                for i in range(len(coords) - 1):
                    line = LineString([coords[i], coords[i+1]])
                    all_lines.append(line)
    
    # Step 3: Create nodes at all vertex positions
    nodes = {}
    for (x, y), node_id in vertex_id_map.items():
        config = np.array([x, y])
        node = graph.add_node(config)
        # Set the node's idx to match our ordered numbering
        node.idx = node_id
        nodes[(x, y)] = node
    
    # Step 4: Create edges between nodes
    merged_geometry = unary_union(all_lines)
    edges_added = set()
    
    if isinstance(merged_geometry, LineString):
        process_line_for_edges(merged_geometry, nodes, graph, edges_added, velocity, min_distance)
    else:  # MultiLineString
        for line in merged_geometry.geoms:
            process_line_for_edges(line, nodes, graph, edges_added, velocity, min_distance)
    
    print(f"Created temporal graph with {graph.get_nodes_length()} nodes and {graph.get_edges_length()} edges")
    
    # Convert NumPy float values to Python floats
    for node in graph.nodes:
        if isinstance(node.config, np.ndarray):
            node.config = [float(x) for x in node.config]
    
    return graph

def process_line_for_edges(line, nodes, graph, edges_added, velocity, min_distance):
    """
    Process a line by creating graph edges between nodes on the line.
    """
    # Find nodes that lie on this line
    line_nodes = []
    for (x, y), node in nodes.items():
        point = Point(x, y)
        if point.distance(line) < min_distance:
            # This node is on or very close to the line
            line_nodes.append((node, line.project(point)))
    
    # Sort nodes by their position along the line
    line_nodes.sort(key=lambda x: x[1])
    
    # Create edges between consecutive nodes
    for i in range(len(line_nodes) - 1):
        node1 = line_nodes[i][0]
        node2 = line_nodes[i+1][0]
        
        # Skip if nodes are the same
        if node1.idx == node2.idx:
            continue
        
        # Create a unique key for this edge
        edge_key = tuple(sorted([node1.idx, node2.idx]))
        
        # Skip if we've already added this edge
        if edge_key in edges_added:
            continue
        
        # Calculate distance and travel time
        distance = np.linalg.norm(node1.config - node2.config)
        
        # Skip if distance is too small
        if distance < min_distance:
            continue
        
        travel_time = math.ceil(distance / velocity)
        travel_time = max(1, travel_time)  # Ensure minimum travel time of 1
        
        # Create a simple solution object for the trajectory
        solution = {
            'cost': distance,
            'path': np.array([node1.config, node2.config])
        }
        
        # Add edge to the graph
        graph.add_edge(node1, node2, travel_time, solution)
        edges_added.add(edge_key)

def visualize_temporal_graph(graph, map_width, map_height, show_node_indices=False, show_travel_times=False):
    """
    Visualize the temporal graph.
    
    Parameters:
    - graph: TemporalGraph object
    - map_width, map_height: Dimensions of the map
    - show_node_indices: Whether to show node indices
    - show_travel_times: Whether to show travel times on edges
    """
    plt.figure(figsize=(12, 12))
    
    # Plot nodes
    for node in graph.nodes:
        x, y = node.config
        plt.plot(x, y, 'o', color='blue', markersize=6)
        
        if show_node_indices:
            plt.text(x, y+0.5, f"{node.idx}", fontsize=8, 
                     horizontalalignment='center', verticalalignment='bottom')
    
    # Plot edges
    for edge_idx, (start_idx, end_idx, travel_time) in enumerate(graph.edges):
        start_node = graph.nodes[start_idx]
        end_node = graph.nodes[end_idx]
        
        start_x, start_y = start_node.config
        end_x, end_y = end_node.config
        
        plt.plot([start_x, end_x], [start_y, end_y], 'k-', linewidth=1.0)
        
        if show_travel_times:
            # Place the travel time text at the middle of the edge
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Offset the text slightly
            offset = 0.3
            dx = end_x - start_x
            dy = end_y - start_y
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 0:
                dx, dy = -dy/norm * offset, dx/norm * offset
            
            plt.text(mid_x + dx, mid_y + dy, f"{travel_time}", fontsize=7, 
                     horizontalalignment='center', verticalalignment='center')
    
    plt.title(f"Temporal Graph: {graph.get_nodes_length()} nodes, {graph.get_edges_length()} edges")
    plt.xlim(0, map_width)
    plt.ylim(0, map_height)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_graph_with_arrows(graph, map_width, map_height, node_size=50, arrow_size=10, 
                               show_node_indices=True, show_travel_times=False, 
                               show_all_edges=True):
    """
    Visualize the temporal graph with arrows showing edge directions.
    
    Parameters:
    - graph: TemporalGraph object
    - map_width, map_height: Dimensions of the map
    - node_size: Size of node markers
    - arrow_size: Size of arrow markers
    - show_node_indices: Whether to show node indices
    - show_travel_times: Whether to show travel times on edges
    - show_all_edges: If False, only show one direction for each edge
    """
    plt.figure(figsize=(12, 12))
    
    # Create a set to track bidirectional edges we've already plotted, if not showing all edges
    plotted_edges = set() if not show_all_edges else None
    
    # Plot edges with arrows
    for edge_idx, (start_idx, end_idx, travel_time) in enumerate(graph.edges):
        # Skip duplicate bidirectional edges if not showing all
        if not show_all_edges:
            edge_pair = tuple(sorted([start_idx, end_idx]))
            if edge_pair in plotted_edges:
                continue
            plotted_edges.add(edge_pair)
        
        start_node = graph.nodes[start_idx]
        end_node = graph.nodes[end_idx]
        
        start_x, start_y = start_node.config
        end_x, end_y = end_node.config
        
        # Calculate the edge vector
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Draw the edge as an arrow
        plt.arrow(start_x, start_y, dx*0.9, dy*0.9, 
                 head_width=arrow_size/10, head_length=arrow_size/8, 
                 fc='blue', ec='blue', alpha=0.6,
                 length_includes_head=True)
        
        # Show travel time
        if show_travel_times:
            # Position the text at the middle of the edge
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Get perpendicular vector for text offset
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                # Offset text perpendicular to the edge
                offset = max(map_width, map_height) / 100  # Scale with map size
                ndx, ndy = -dy/length * offset, dx/length * offset
                plt.text(mid_x + ndx, mid_y + ndy, f"{travel_time}", 
                         fontsize=8, ha='center', va='center', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Plot nodes
    node_x = [node.config[0] for node in graph.nodes]
    node_y = [node.config[1] for node in graph.nodes]
    plt.scatter(node_x, node_y, s=node_size, c='red', zorder=10)
    
    # Add node indices
    if show_node_indices:
        for node in graph.nodes:
            x, y = node.config
            plt.text(x, y, f"{node.idx}", fontsize=9, ha='center', va='center', 
                    color='white', fontweight='bold', zorder=11)
    
    plt.title(f"Temporal Graph: {graph.get_nodes_length()} nodes, {graph.get_edges_length()} edges")
    plt.xlim(0, map_width)
    plt.ylim(0, map_height)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

# Create the map
map_width, map_height = 50, 50
irregular_polygons = generate_weighted_voronoi_polygons(map_width, map_height, num_points=50, rand_seed=128)
# map_width, map_height = 100, 100
# irregular_polygons = generate_weighted_voronoi_polygons(map_width, map_height, num_points=100, rand_seed=42)
shrunk_polygons = shrink_polygons(irregular_polygons, shrink_distance=0.5)
visualize_map_detailed(irregular_polygons, shrunk_polygons, map_width, map_height)

# Convert to temporal graph
graph = convert_polygons_to_temporal_graph(irregular_polygons, velocity=5.0)

filename = "50_graph.pkl"

save_data = {
    'graph': graph,
    'polygons': irregular_polygons,
    'shrunk_polygons': shrunk_polygons,
    'map_width': map_width,
    'map_height': map_height
}

with open(filename, 'wb') as f:
    pickle.dump(save_data, f)

print(f"Map data saved to {filename}")

# Visualize the temporal graph
visualize_temporal_graph(graph, map_width, map_height, show_node_indices=True, show_travel_times=False)
