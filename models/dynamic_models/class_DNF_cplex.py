import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cost_dir = os.path.join(current_dir, 'costs')

import numpy as np
from utils.temporal_graph import Node, TemporalGraph
from termcolor import colored
from docplex.mp.model import Model as cplex_Model
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Tuple, Any

class dynamic_nf:
    
    @classmethod
    def from_trajectories(cls, model, dnf_config: Dict) -> 'dynamic_nf':
        """Create DNF instance from experiment configuration in JSON.
        
        Args:
            model: CPLEX optimization model
            dnf_config: Dictionary containing network flow configuration from JSON
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
        cost_move = []

        trajectories = list(trajectory_data['trajectories'].values())
        dT_traj = trajectory_data.get('dT_traj')
        
        for trajectory in trajectories:
            begin_point = {'com_x': round(float(trajectory['com'][0, 0]), 3), 'com_y': round(float(trajectory['com'][0, 1]), 3)}
            end_point = {'com_x': round(float(trajectory['com'][-1, 0]), 3),'com_y': round(float(trajectory['com'][-1, 1]), 3)}
            
            node_A = graph.add_node(begin_point['com_x'], begin_point['com_y'])
            node_B = graph.add_node(end_point['com_x'], end_point['com_y'])
            
            traj_duration = (len(trajectory['theta'])-1) * dT_traj
            travel_time = int(np.ceil(traj_duration / dT_network).item())

            movement_cost = trajectory.get('cost', 1.0)
            cost_move.extend([movement_cost, movement_cost])
            
            graph.add_edge(node_A, node_B, travel_time, trajectory)

        source_nodes = {}
        for pos in source_positions:
            node_idx = graph.find_node_idx(pos['com_x'], pos['com_y'])
            source_nodes[node_idx] = 1
        
        cost_hold = np.load(cost_files['hold'])
        cost_move = np.load(cost_files['move'])

        return cls(model=model, nf_name=nf_name, N=N, graph=graph, cost_move=cost_move, cost_hold=cost_hold, node_ini=source_nodes)

    def __init__(self, model, nf_name, N, graph, cost_move, cost_hold):
        """Initialize Dynamic Network Flow model for CPLEX.

        Args:
            model: CPLEX optimization model
            nf_name: Name identifier for this network flow
            N: Total number of timesteps in planning horizon
            graph: Graph object containing nodes and edges
            cost_move: Array of movement costs
            cost_hold: Array of holding costs
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

        # Print
        print("======================================================================================")
        print(colored("Building dynamic network flow: {} with CPLEX".format(self.nf_name), 'green'))
        print(colored('Time horizon: {}, number of section is {}'.format(self.N, self.N_sec), 'green'))
        print(colored('Number of nodes per time-step: {}'.format(self.nb), 'green'))
        print(colored('Number of mobile edges per time-step: {}'.format(self.n_mobile_edge), 'green'))

    def to_networkx(self):
        """Convert Graph structure to NetworkX for visualization"""
        G = nx.DiGraph()
        
        for node in self.nodes:
            G.add_node(node.idx, 
                      pos={'com_x': node.com_x, 'com_y': node.com_y},
                      cost=self.cost_hold[node.idx])
        
        for i, edge in enumerate(self.edges):
            G.add_edge(edge[0], edge[1], 
                      travel_time=edge[2],
                      cost=self.cost_move[i])
            
        return G
         
    def setup_model(self):
        """Setup variables for CPLEX"""
        # ys variables
        self.ys = []
        for i in range(self.nb):
            var = self.m.continuous_var(lb=0.0, ub=1.0, name=f'{self.nf_name}_ys_{i}')
            self.ys.append(var)
        
        # ye_hold variables
        self.ye_hold = {}
        for i in range(self.nb):
            for j in range(self.N_sec):
                var = self.m.continuous_var(lb=0.0, ub=1.0, name=f'{self.nf_name}_ye_hold_{i}_{j}')
                self.ye_hold[i, j] = var
        
        # ye variables (movement edges)
        self.ye = []
        for i_m in range(self.n_mobile_edge):
            ne = self.N - self.edges[i_m][2]
            ye_edge = []
            for j in range(ne):
                var = self.m.continuous_var(lb=0.0, ub=1.0, name=f'{self.nf_name}_ye_{i_m}_{j}')
                ye_edge.append(var)
            self.ye.append(ye_edge)

    def set_objective(self):
        """Set objective for CPLEX"""
        obj_terms = []

        # Cost of hold
        for nn in range(self.N_sec):
            for vv in range(self.nb):
                obj_terms.append(self.cost_hold[vv, nn] * self.ye_hold[vv, nn])

        # Cost of moving
        for ii, ed in enumerate(self.ye):
            for tt in range(len(ed)):
                obj_terms.append(self.cost_move[ii, tt] * self.ye[ii][tt])

        if obj_terms:
            self.obj = self.m.sum(obj_terms)
        else:
            # Fallback if no terms
            self.obj = self.m.sum(self.ys[i] for i in range(self.nb)) * 0

    def enforce_flow_conservation_and_degree(self):
        """Enforce constraints for CPLEX"""
        for nn in range(self.N_sec):  
            for vv in range(self.nb):
                # Build LHS (incoming flow)
                lhs_terms = []
                
                if nn == 0:
                    lhs_terms.append(self.ys[vv])
                else:
                    lhs_terms.append(self.ye_hold[vv, nn-1])
                    for ed_in in self.in_edges[vv]:
                        if (nn - self.edges[ed_in][2] >= 0):
                            lhs_terms.append(self.ye[ed_in][nn - self.edges[ed_in][2]])

                # Build RHS (outgoing flow)
                rhs_terms = [self.ye_hold[vv, nn]]
                for ed_out in self.out_edges[vv]:
                    if (nn + self.edges[ed_out][2] <= self.N_sec):
                        rhs_terms.append(self.ye[ed_out][nn])

                # Create expressions
                lhs_expr = self.m.sum(lhs_terms) if len(lhs_terms) > 1 else lhs_terms[0]
                rhs_expr = self.m.sum(rhs_terms) if len(rhs_terms) > 1 else rhs_terms[0]

                # Add constraints
                self.m.add_constraint(lhs_expr == rhs_expr, ctname=f"flow_conservation_{nn}_{vv}")
                self.m.add_constraint(lhs_expr <= 1.0, ctname=f"degree_constraint_{nn}_{vv}")

    def set_initial_condition(self, node_ini):
        """Enforce initial condition for CPLEX"""
        self.node_ini = node_ini

        for ii in range(self.nb):
            if not ii in self.node_ini.keys():
                self.m.add_constraint(self.ys[ii] == 0, ctname=f"initial_zero_{ii}")
            else:
                self.m.add_constraint(self.ys[ii] == self.node_ini[ii], ctname=f"initial_value_{ii}")
        
    def setup_problem(self):
        """Setup the complete optimization problem"""
        self.setup_model()
        print("Enforcing flow conservation and degree<1 ...")
        self.enforce_flow_conservation_and_degree()
        print("Setting objectives ...")
        self.set_objective()

    def validate_edges(self):
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
        
        G = self.to_networkx()
        
        pos = {node: (data['pos']['com_x'], data['pos']['com_y']) 
               for node, data in G.nodes(data=True)}
        
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=500)
        
        if hasattr(self, 'node_ini') and self.node_ini:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=list(self.node_ini.keys()),
                                 node_color='lightgreen',
                                 node_size=500)

        edge_labels = {(u,v): f't={d["travel_time"]}'
                      for u,v,d in G.edges(data=True)}
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        labels = {n: f"{n}\n({data['pos']['com_x']:.1f},\n{data['pos']['com_y']:.1f})"
                 for n, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels)
        
        plt.title(f"DNF Graph: {self.nf_name}")
        plt.axis('off')
        plt.show()

    def visualize_dynamic_flow(self, figsize=(15, 10)):
        """Visualize the time-expanded network."""
        plt.figure(figsize=figsize)
        
        G = nx.DiGraph()
        
        source = ('s', -1)
        G.add_node(source, layer=-1)
        
        for t in range(self.N):
            for node in self.nodes:
                G.add_node((node.idx, t), layer=t)
                
                if t < self.N - 1:
                    G.add_edge((node.idx, t), (node.idx, t+1), 
                             edge_type='hold')
        
        for edge in self.edges:
            start_idx, end_idx, travel_time = edge
            for t in range(self.N - travel_time):
                G.add_edge((start_idx, t), (end_idx, t + travel_time),
                          edge_type='move')
        
        if hasattr(self, 'node_ini'):
            for start_node in self.node_ini:
                G.add_edge(source, (start_node, 0), edge_type='source')
            
        pos = {}
        pos[source] = (-1, self.nb/2)
        for node in G.nodes():
            if node != source:
                n, t = node
                pos[node] = (t, n)
        
        edges_source = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='source']
        edges_hold = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='hold']
        edges_move = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='move']
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_source, edge_color='green', style='solid', arrowsize=15)
        nx.draw_networkx_edges(G, pos, edgelist=edges_hold, edge_color='black', style='solid', arrowsize=10)
        nx.draw_networkx_edges(G, pos, edgelist=edges_move, edge_color='red', style='dashed', arrowsize=10)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='lightgreen', node_shape='s', node_size=500)
        
        labels = {n: f"{n[0]}" if n != source else 's' for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Time-expanded Network Flow: {self.nf_name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Node ID")
        plt.grid(True)
        plt.show()
        
