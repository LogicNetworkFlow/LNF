import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cost_dir = os.path.join(current_dir, 'costs')

import numpy as np
from utils.temporal_graph import Node, TemporalGraph
from termcolor import colored
import gurobipy as go
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple, Any

class dynamic_nf:

    def __init__(self, model, nf_name, N, graph, cost_move, cost_hold):
        """Initialize Dynamic Network Flow model.

        Args:
            model: Optimization model (e.g., Gurobi model)
            nf_name: Name identifier for this network flow
            N: Total number of timesteps in planning horizon
            nb: Number of nodes (locations) in the network
            edges: List of [start_idx, end_idx, travel_time] representing possible movements where travel_time is the number of timesteps needed to traverse the edge
            cost_move: Array of movement costs (length = number of edges x (planning_horizon-1))
                       Can come from trajectory optimization, terrain difficulty, energy consumption, safety considerations, etc.
            cost_hold: Array of holding costs (length = number of nodes x (planning_horizon-1))
                      Cost for staying at each location, can represent danger level, exposure to elements, resource consumption, etc.

        Network Flow Structure:
            N: Planning horizon length (e.g., 40 timesteps)
            N_sec: Number of sections (N-1) where flows can occur, excluding the initial flow from the source
            nb: Number of base nodes (locations) where units can be
            edges: Movement possibilities between nodes
            
        Cost Structure:
            cost_move: Cost for traversing each edge, indexed same as edges list
            cost_hold: Cost for staying at each node, indexed by node number
        """

        self.m = model
        self.nf_name = nf_name

        self.N = N
        self.N_sec = N-1
        
        self.graph = graph
        
        # Get graph properties
        self.nodes = graph.get_nodes_list()
        self.edges = graph.get_edges_list()
        self.nb = graph.get_nodes_length()
        self.n_mobile_edge = graph.get_edges_length()
        self.in_edges = {node.idx: node.in_edges for node in self.nodes}
        self.out_edges = {node.idx: node.out_edges for node in self.nodes}

        self.cost_hold = cost_hold
        self.cost_move = cost_move

        # self.cost_hold = np.random.uniform(0, 1, len(self.nodes))
        # self.cost_move = np.random.uniform(0, 1, len(self.edges))

        # Create 2D arrays with random values between 0 and 1 for each node/edge at each time step
        # self.cost_hold = np.random.uniform(0, 1, (len(self.nodes), self.N_sec))
        # self.cost_move = np.random.uniform(0, 1, (len(self.edges), self.N_sec))

        # np.save('dynamic_models/costs/0_10robot_biped_cost_hold_TV.npy', self.cost_hold)
        # np.save('dynamic_models/costs/0_10robot_biped_cost_move_TV.npy', self.cost_move)

        # self.cost_hold = np.load('/home/x/Desktop/LDF/dynamic_models/costs/0_10robot_biped_cost_hold_TV.npy')
        # self.cost_move = np.load('/home/x/Desktop/LDF/dynamic_models/costs/0_10robot_biped_cost_move_TV.npy')

        # Verify that edges do not repeat
        # assert not has_duplicate_first_two(self.edges)

        # Print ----------------------------------------------------------------------------------
        print("======================================================================================")
        print(colored("Building dynamic network flow: {} ".format(self.nf_name), 'green'))
        print(colored('Time horizon: {}, number of section is {}'.format(self.N, self.N_sec), 'green'))
        print(colored('Number of nodes per time-step: {}'.format(self.nb), 'green'))
        print(colored('Number of mobile edges per time-step: {}'.format(self.n_mobile_edge), 'green'))

    def to_networkx(self):
        """Convert Graph structure to NetworkX for visualization"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(node.idx, 
                      pos={'com_x': node.com_x, 'com_y': node.com_y},
                      cost=self.cost_hold[node.idx])
        
        # Add edges
        for i, edge in enumerate(self.edges):
            G.add_edge(edge[0], edge[1], 
                      travel_time=edge[2],
                      cost=self.cost_move[i])
            
        return G
         
    def setup_model(self):
        self.ys =      self.m.addMVar((self.nb,           ), lb=0.0, ub=1.0, name=self.nf_name+'_ys')
        self.ye_hold = self.m.addMVar((self.nb, self.N_sec), lb=0.0, ub=1.0, name=self.nf_name+'_ye_hold')
        self.ye = []
        for i_m in range(self.n_mobile_edge):
            # For example, if NF has N=3, an edge travel 1 section, there will be 2 such edges.
            #   N:           0         1         2
            #              site1 --- site1 --- site1
            #   source --  site2 --- site2 --- site2
            #              site3 --- site3 --- site3
            ne = self.N - self.edges[i_m][2]  
            self.ye.append(self.m.addMVar((ne, ), lb=0.0, ub=1.0, name=self.nf_name+'_ye'))

    def set_objective(self):
        self.obj = 0.0

        # # Cost of hold
        # for nn in range(self.N_sec):
        #     for vv in range(self.nb):
        #         self.obj += self.ye_hold[vv, nn]*self.cost_hold[vv]
        
        # # Cost of moving
        # for ii, ed in enumerate(self.ye):
        #     for tt in range(ed.shape[0]):
        #         self.obj += self.ye[ii][tt]*self.cost_move[ii]
        
        # Cost of hold - now using time-dependent random costs
        for nn in range(self.N_sec):
            for vv in range(self.nb):
                self.obj += self.ye_hold[vv, nn] * self.cost_hold[vv, nn]

        # Cost of moving - now using time-dependent random costs
        for ii, ed in enumerate(self.ye):
            for tt in range(ed.shape[0]):
                self.obj += self.ye[ii][tt] * self.cost_move[ii, tt]

    # def enforce_unit_flow(self):
    #     self.m.addConstr(self.ys[:].sum() == len(self.node_ini))

    def enforce_flow_conservation_and_degree(self):
        # The target vertex flow conservation constraint is trivial and omitted
        for nn in range(self.N_sec):  
            for vv in range(self.nb):
                if nn==0:
                    lhs = self.ys[vv]
                else:
                    lhs = self.ye_hold[vv, nn-1]
                    for ed_in in self.in_edges[vv]:
                        if (nn - self.edges[ed_in][2] >= 0):
                            lhs += self.ye[ed_in][nn - self.edges[ed_in][2]]

                rhs = self.ye_hold[vv, nn]
                for ed_out in self.out_edges[vv]:
                    if (nn + self.edges[ed_out][2] <= self.N_sec):
                        rhs += self.ye[ed_out][nn]

                self.m.addConstr(lhs == rhs)
                self.m.addConstr(lhs <= 1.0)  # Enforce degree constraint for collision avoidance

    def set_initial_condition(self, node_ini):
        """Enforce initial condition.
        Args:

            node_ini: Dictionary mapping source node indices to their initial flow (typically 1)
                      e.g., {0: 1, 5: 1} means units start at nodes 0 and 5
        """

        self.node_ini = node_ini

        for ii in range(self.nb):
            if not ii in self.node_ini.keys():
                self.ys[ii].lb = self.ys[ii].ub = 0
            else:
                self.ys[ii].lb = self.ys[ii].ub = self.node_ini[ii]
        
    def setup_problem(self):

        # self.validate_edges()
        self.setup_model()
        print("Enforcing flow conservation and degree<1 ...")
        self.enforce_flow_conservation_and_degree()
        print("Setting objectives ...")
        self.set_objective()
        self.m.update()
        
        # self.m.write("NF.mps")

        # print("Loading model ...")
        # self.m = go.read("NF.mps")

    def validate_edges(self):
        # To validate connection graph, print in/out edges for each node
        for ii in range(self.nb):
            print("==========================================")
            print("node {}".format(ii))
            print("Out edges")
            for item in self.out_edges[ii]:
                print("Edge {} ID {}".format(self.edges[item], item))
            print("In edges")
            for item in self.in_edges[ii]:
                print("Edge {} ID {}".format(self.edges[item], item))

    def visualize_graph(self, figsize=(10, 10)):
        """Visualize the DNF graph structure using networkx."""
        plt.figure(figsize=figsize)
        
        # Convert to NetworkX graph
        G = self.to_networkx()
        
        # Get node positions for layout
        pos = {node: (data['pos']['com_x'], data['pos']['com_y']) 
               for node, data in G.nodes(data=True)}
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=500)
        
        # Highlight source nodes
        if self.node_ini:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=list(self.node_ini.keys()),
                                 node_color='lightgreen',
                                 node_size=500)

        # Draw edges with travel time labels
        edge_labels = {(u,v): f't={d["travel_time"]}'
                      for u,v,d in G.edges(data=True)}
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        # Add node labels with coordinates
        labels = {n: f"{n}\n({data['pos']['com_x']:.1f},\n{data['pos']['com_y']:.1f})"
                 for n, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels)
        
        plt.title(f"DNF Graph: {self.nf_name}")
        plt.axis('off')
        plt.show()

    def visualize_dynamic_flow(self, figsize=(15, 10)):
        """Visualize the time-expanded network."""
        plt.figure(figsize=figsize)
        
        # Create expanded digraph
        G = nx.DiGraph()
        
        # Add source node
        source = ('s', -1)
        G.add_node(source, layer=-1)
        
        # Add time-expanded nodes
        for t in range(self.N):
            for node in self.nodes:
                G.add_node((node.idx, t), layer=t)
                
                # Add hold edges if not last timestep
                if t < self.N - 1:
                    G.add_edge((node.idx, t), (node.idx, t+1), 
                             edge_type='hold')
        
        # Add movement edges
        for edge in self.edges:
            start_idx, end_idx, travel_time = edge
            for t in range(self.N - travel_time):
                G.add_edge((start_idx, t), (end_idx, t + travel_time),
                          edge_type='move')
        
        # Add source edges
        for start_node in self.node_ini:
            G.add_edge(source, (start_node, 0), edge_type='source')
            
        # Create layout
        pos = {}
        # Position source node
        pos[source] = (-1, self.nb/2)
        # Position other nodes in a grid
        for node in G.nodes():
            if node != source:
                n, t = node
                pos[node] = (t, n)
        
        # Draw different edge types with different styles
        edges_source = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='source']
        edges_hold = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='hold']
        edges_move = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='move']
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_source, edge_color='green', style='solid', arrowsize=15)
        nx.draw_networkx_edges(G, pos, edgelist=edges_hold, edge_color='black', style='solid', arrowsize=10)
        nx.draw_networkx_edges(G, pos, edgelist=edges_move, edge_color='red', style='dashed', arrowsize=10)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='lightgreen', node_shape='s', node_size=500)
        
        # Add labels
        labels = {n: f"{n[0]}" if n != source else 's' for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Time-expanded Network Flow: {self.nf_name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Node ID")
        plt.grid(True)
        plt.show()

    # ========================================================================================================================
    @classmethod
    def from_trajectories(cls, model, dnf_config: Dict) -> 'dynamic_nf':
        """Create DNF instance from experiment configuration in JSON.
        
        Args:
            model: Optimization model
            dnf_config: Dictionary containing network flow configuration from JSON
                Expected format:
                {
                    "nf_name": str,
                    "trajectory_library_file": str,
                    "N": int,
                    "dT_network": float,
                    "source_positions": List[Dict],
                    "cost_files": {
                        "move": str,
                        "hold": str
                    }
                }
        """
        nf_name = dnf_config['nf_name']
        N = dnf_config['N']
        dT_network = dnf_config['dT_network']
        source_positions = dnf_config['source_positions']
        cost_files = dnf_config['cost_files']

        # Load trajectory data
        with open(dnf_config['trajectory_library_file'], 'rb') as f:
            trajectory_data = pickle.load(f)
        
        graph = TemporalGraph()
        # TODO: can we place cost_move and cost_hold into the graph class?
        cost_move = []  # Store costs in order of edge addition - perhaps we can move this into the graph class!

        trajectories = list(trajectory_data['trajectories'].values())
        dT_traj = trajectory_data.get('dT_traj')
        
        for trajectory in trajectories:
            # Extract start and end points
            begin_point = {'com_x': round(float(trajectory['com'][0, 0]), 3), 'com_y': round(float(trajectory['com'][0, 1]), 3)}
            end_point = {'com_x': round(float(trajectory['com'][-1, 0]), 3),'com_y': round(float(trajectory['com'][-1, 1]), 3)}
            
            node_A = graph.add_node(begin_point['com_x'], begin_point['com_y'])
            node_B = graph.add_node(end_point['com_x'], end_point['com_y'])
            
            # This effectively enforce: time_scaling_ratio = dT_network / dT_traj
            traj_duration = (len(trajectory['theta'])-1) * dT_traj
            travel_time = int(np.ceil(traj_duration / dT_network).item())

            # Get movement cost
            movement_cost = trajectory.get('cost', 1.0)  # Default to 1.0 if no cost
            cost_move.extend([movement_cost, movement_cost])  # Add twice for bidirectional edges
            
            graph.add_edge(node_A, node_B, travel_time, trajectory)

        # Create source nodes dictionary
        source_nodes = {}
        for pos in source_positions:
            node_idx = graph.find_node_idx(pos['com_x'], pos['com_y'])
            source_nodes[node_idx] = 1
        
        cost_hold = np.load(cost_files['hold'])
        cost_move = np.load(cost_files['move'])

        # Create instance
        return cls(model=model, nf_name=nf_name, N=N, graph=graph, cost_move=cost_move, cost_hold=cost_hold, node_ini=source_nodes)
    
