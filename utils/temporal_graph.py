import numpy as np

class Node:
    def __init__(self, config, idx, name=None):
        self.name = name  # For searching node_id
        self.config = np.asarray(config)  # Configuration vector (position, angles, etc.)
        self.idx = idx
        self.out_edges = []
        self.in_edges = []

    def __repr__(self):
        name_str = f", name={self.name}" if self.name else ""
        return f"Node(config={self.config}{name_str})"

    def __eq__(self, other):
        """Override equality to compare nodes based on their configurations."""
        return isinstance(other, Node) and np.array_equal(self.config, other.config)
    
    def add_out_edge(self, edge_idx):
        """record the edge idx that flow out from node"""
        self.out_edges.append(edge_idx)
    
    def add_in_edge(self, edge_idx):
        """Record the edge idx that flow into this node"""
        self.in_edges.append(edge_idx)

class TemporalGraph:
    def __init__(self):
        self.nodes = []  # List of nodes (node objects)
        self.edges = []  # List of edges (tuple: (idx of start_node, idx of end_node, travel_time))
        self.lib_traj = {}  # Dictionary to store trajectories

    def add_node(self, config, name=None):
        """Checks if a node with the given configuration exists, and adds it if not."""
        node = Node(config, len(self.nodes), name)
        
        # Check if the node already exists in the list
        for existing_node in self.nodes:
            if existing_node == node:
                return existing_node  # Return the existing node if it's already in the list

        # Node doesn't exist, so add it
        self.nodes.append(node)
        return node

    def add_edge(self, node_A, node_B, travel_time, sol):
        """Adds an edge between node_A and node_B with a given travel time and trajectory."""
        # Create keys for both directions
        forward_key = f"{node_A.idx}-{node_B.idx}"
        reverse_key = f"{node_B.idx}-{node_A.idx}"
        
        # Check if either direction exists
        if not any((node_A == start and node_B == end) or (node_B == start and node_A == end) for start, end, _ in self.edges):
            # Add forward edge
            self.edges.append([node_A.idx, node_B.idx, travel_time])
            node_A.add_out_edge(len(self.edges)-1)
            node_B.add_in_edge(len(self.edges)-1)
            self.lib_traj[forward_key] = sol

            # Add reverse edge
            self.edges.append([node_B.idx, node_A.idx, travel_time])
            node_A.add_in_edge(len(self.edges)-1)
            node_B.add_out_edge(len(self.edges)-1)
            
            # Check if reverse trajectory exists in solution
            if reverse_key in sol:  # If trajectory library stored reverse
                self.lib_traj[reverse_key] = sol[reverse_key]
            else:  # If not, create reversed trajectory
                flipped_sol = {}
                for key, val in sol.items():
                    if isinstance(val, np.ndarray):
                        flipped_sol[key] = np.flip(val, axis=0)
                    else:
                        flipped_sol[key] = val
                self.lib_traj[reverse_key] = flipped_sol
        else:
            print(f"Edge between {node_A} and {node_B} or reverse already exists. Skipping addition.")
        
    def find_node_by_config(self, config):
        """Finds the node with the given configuration."""
        config_array = np.asarray(config)
        for node in self.nodes:
            if np.array_equal(node.config, config_array):
                return node
        print(f"No node found with configuration {config}")
        return None
    
    def find_node_idx_by_config(self, config):
        """Finds the index of the node with the given configuration."""
        node = self.find_node_by_config(config)
        return node.idx if node else None
    
    def find_node_by_name(self, search_name):
        """Finds a node with the given name."""
        for node in self.nodes:
            if node.name and search_name in node.name:
                return node
        return None
    
    def get_nodes_list(self):
        """Return the list of all nodes"""
        return self.nodes
    
    def get_edges_list(self):
        """Return the list of all edges"""
        return self.edges
    
    def get_nodes_length(self):
        """Return the number of nodes"""
        return len(self.nodes)
    
    def get_edges_length(self):
        """Return the number of edges"""
        return len(self.edges)
    
    def get_lib_trajectory(self):
        """Return lib traj"""
        return self.lib_traj
    