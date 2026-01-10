import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.logic_models.class_LNF import logic_nf
from models.logic_models.class_LT import logic_tree
from models.logic_models.STL.predicate import ConvexSetPredicate
from models.logic_models.STL.data_structure_conversion import assemble_logic_nf_info, assemble_logic_tree_info

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
from termcolor import colored


@dataclass
class PointMassParams:
    """Parameters for point mass dynamics and environment"""
    nx: int  # state dimension [x, y, xdot, ydot]
    nu: int  # control dimension [force_x, force_y]
    N: int  # horizon length
    dt: float  # time step
    mass: float  # mass of the point
    regions: List[Dict]  # Region limits
    region_control_limits: List[Dict]  # Region-specific control limits
    region_costs: List[float]  # Region-specific costs
    nz: int  # number of regions
    M: float  # big M constant
    max_velocity: float  # maximum velocity


class PointMassDynamics:
    """Point mass dynamics with environment constraints"""
    
    def __init__(self, params: PointMassParams):
        self.p = params
        
        # Discrete-time dynamics: x_{k+1} = A*x_k + B*u_k
        self.A = np.eye(self.p.nx)
        self.A[0, 2] = self.p.dt  # x += xdot * dt
        self.A[1, 3] = self.p.dt  # y += ydot * dt
        
        self.B = np.zeros((self.p.nx, self.p.nu))
        self.B[2, 0] = self.p.dt / self.p.mass  # xdot += force_x * dt / mass
        self.B[3, 1] = self.p.dt / self.p.mass  # ydot += force_y * dt / mass

    def setup_optimization(self, model, x0: np.ndarray):
        """Setup optimization problem with Big-M formulation"""
        self.model = model
        
        # Create variables
        x = {}  # States
        u = {}  # Controls
        z = {}  # Binary variables for regions
        
        # State variables
        for t in range(self.p.N + 1):
            for i in range(self.p.nx):
                x[t,i] = self.model.addVar(lb=-self.p.M, ub=self.p.M, name=f'x_{t}_{i}')
        
        # Control variables
        for t in range(self.p.N):
            for i in range(self.p.nu):
                u[t,i] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'u_{t}_{i}')
        
        # Binary variables for regions
        for t in range(self.p.N + 1):
            for r in range(self.p.nz):
                z[t,r] = self.model.addVar(vtype=GRB.BINARY, name=f'z_{t}_{r}')
        
        # Initial state constraints
        for i in range(self.p.nx):
            self.model.addConstr(x[0,i] == x0[i])
        
        # Dynamics constraints
        for t in range(self.p.N):
            for i in range(self.p.nx):
                expr = gp.LinExpr()
                for j in range(self.p.nx):
                    expr.add(x[t,j], self.A[i,j])
                for j in range(self.p.nu):
                    expr.add(u[t,j], self.B[i,j])
                self.model.addConstr(x[t+1,i] == expr)

        # Velocity constraints
        for t in range(self.p.N + 1):
            self.model.addConstr(x[t,2] <= self.p.max_velocity)
            self.model.addConstr(x[t,2] >= -self.p.max_velocity)
            self.model.addConstr(x[t,3] <= self.p.max_velocity)
            self.model.addConstr(x[t,3] >= -self.p.max_velocity)
        
        # Region constraints using Big-M
        for t in range(self.p.N + 1):
            # Exactly one region at a time
            self.model.addConstr(sum(z[t,r] for r in range(self.p.nz)) == 1)
            
            # Position constraints for each region
            for r, region in enumerate(self.p.regions):
                self.model.addConstr(x[t,0] <= region['xu'] + self.p.M * (1 - z[t,r]))
                self.model.addConstr(x[t,0] >= region['xl'] - self.p.M * (1 - z[t,r]))
                self.model.addConstr(x[t,1] <= region['yu'] + self.p.M * (1 - z[t,r]))
                self.model.addConstr(x[t,1] >= region['yl'] - self.p.M * (1 - z[t,r]))
        
        # Region-specific control limits
        for t in range(self.p.N):
            for r in range(self.p.nz):
                limits = self.p.region_control_limits[r]
                self.model.addConstr(u[t,0] <= limits['max_force_x'] + self.p.M * (1 - z[t,r]))
                self.model.addConstr(u[t,0] >= -limits['max_force_x'] - self.p.M * (1 - z[t,r]))
                self.model.addConstr(u[t,1] <= limits['max_force_y'] + self.p.M * (1 - z[t,r]))
                self.model.addConstr(u[t,1] >= -limits['max_force_y'] - self.p.M * (1 - z[t,r]))

        # Objective function
        obj = gp.QuadExpr()
        
        # Region costs
        for t in range(self.p.N + 1):
            for r in range(self.p.nz):
                obj.add(self.p.region_costs[r] * z[t,r])
        
        # Control effort
        state_cost_weight = 0.001
        for t in range(self.p.N):
            for i in range(self.p.nu):
                obj.add(state_cost_weight * u[t,i] * u[t,i])
        
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        
        # Store variables
        self.x_vars = x
        self.u_vars = u
        self.z_vars = z

    def solve_and_extract_trajectory(self):
        """Extract trajectory from solution"""
        if self.model.SolCount > 0:
            x_sol = np.zeros((self.p.N + 1, self.p.nx))
            u_sol = np.zeros((self.p.N, self.p.nu))
            
            for t in range(self.p.N + 1):
                for i in range(self.p.nx):
                    x_sol[t,i] = self.x_vars[t,i].X
                    
            for t in range(self.p.N):
                for i in range(self.p.nu):
                    u_sol[t,i] = self.u_vars[t,i].X
                    
            return x_sol, u_sol
        else:
            print(colored("No feasible solution found", 'red'))
            return None, None

    def plot_trajectory(self, x_traj: np.ndarray, u_traj: np.ndarray, 
                       target_positions=None, save_path=None):
        """Plot trajectory with environment"""
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_facecolor("black")
        
        # Plot regions
        region_color = 'lightgray'
        boundary_color = 'brown'
        
        for region in self.p.regions:
            rectangle = plt.Rectangle(
                (region['xl'], region['yl']), 
                (region['xu'] - region['xl']), 
                (region['yu'] - region['yl']), 
                facecolor=region_color,
                edgecolor=boundary_color,
                linewidth=2,
                zorder=2
            )
            ax.add_patch(rectangle)
        
        # Plot targets
        if target_positions is not None:
            for i, (x_target, y_target) in enumerate(target_positions):
                plt.scatter(x_target, y_target, color='yellow', s=200, marker='*', 
                          edgecolor='black', linewidth=1, zorder=6, 
                          label='Targets' if i == 0 else "")
                plt.text(x_target, y_target + 0.1, r'$\pi_{}$'.format(i), 
                        horizontalalignment='center', verticalalignment='bottom', 
                        fontsize=12, color='black', weight='bold', zorder=7)
        
        # Plot trajectory
        plt.plot(x_traj[:, 0], x_traj[:, 1], 'b-', linewidth=2, zorder=3, label='Trajectory')
        
        # Add direction arrows
        arrow_interval = max(1, len(x_traj) // 8)
        for i in range(0, len(x_traj)-1, arrow_interval):
            dx = x_traj[i+1, 0] - x_traj[i, 0]
            dy = x_traj[i+1, 1] - x_traj[i, 1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                scale = 0.15
                dx_norm = dx / length * scale
                dy_norm = dy / length * scale
                plt.arrow(x_traj[i, 0], x_traj[i, 1], dx_norm, dy_norm,
                        head_width=0.04, head_length=0.06, fc='blue', ec='blue', 
                        zorder=4, alpha=0.8)
        
        # Mark start and end
        plt.scatter(x_traj[0, 0], x_traj[0, 1], color='blue', s=150, marker='o', 
                   label='Start', zorder=5, edgecolor='white', linewidth=2)
        plt.scatter(x_traj[-1, 0], x_traj[-1, 1], color='green', s=150, marker='s', 
                   label='End', zorder=5, edgecolor='white', linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x [m]', color='white')
        plt.ylabel('y [m]', color='white')
        plt.title('Point Mass Navigation', color='white')
        ax.tick_params(colors='white')
        
        # Set limits
        x_min = min(region['xl'] for region in self.p.regions) - 0.3
        x_max = max(region['xu'] for region in self.p.regions) + 0.3
        y_min = min(region['yl'] for region in self.p.regions) - 0.3
        y_max = max(region['yu'] for region in self.p.regions) + 0.3
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
        
        if save_path:
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
            print(f"Figure saved to: {save_path}")
        
        plt.close()


def create_binary_mapping(unique_predicate_dict, z_vars):
    """Map predicates to binary variables"""
    binary_vars_list = []
    for pred_key in unique_predicate_dict.keys():
        parts = pred_key.split('_')
        time_step = int(parts[1][1:])
        region_number = int(parts[2][2:])
        binary_vars_list.append(z_vars[time_step, region_number])
    return np.array(binary_vars_list)


def main():
    print("\n" + "="*80)
    print(colored("Point Mass Navigation with Temporal Logic Constraints", 'cyan', attrs=['bold']))
    print("="*80 + "\n")

    # Problem parameters (from config)
    N = 60
    dt = 0.5
    x0 = np.array([4.39, 3.59, 0.0, 0.0])  # [x, y, xdot, ydot]
    
    # Define regions from config
    regions = [
        {'xl': 0.0, 'xu': 1.0, 'yl': 0.0, 'yu': 5.2},
        {'xl': 1.0, 'xu': 5.0, 'yl': 0.0, 'yu': 1.0},
        {'xl': 2.5, 'xu': 3.5, 'yl': 1.0, 'yu': 1.4},
        {'xl': 5.0, 'xu': 6.0, 'yl': 0.0, 'yu': 5.2},
        {'xl': 1.0, 'xu': 5.0, 'yl': 1.4, 'yu': 2.4},
        {'xl': 1.0, 'xu': 5.0, 'yl': 2.8, 'yu': 3.8},
        {'xl': 1.0, 'xu': 5.0, 'yl': 4.2, 'yu': 5.2},
        {'xl': 2.5, 'xu': 3.5, 'yl': 3.8, 'yu': 4.2},
        {'xl': 1.5, 'xu': 1.5, 'yl': 1.6, 'yu': 1.6},  # Target (point)
        {'xl': 4.8, 'xu': 4.8, 'yl': 0.8, 'yu': 0.8},  # Target (point)
        {'xl': 4.5, 'xu': 4.5, 'yl': 2.0, 'yu': 2.0},  # Target (point)
        {'xl': 2.2, 'xu': 2.2, 'yl': 4.4, 'yu': 4.4},  # Target (point)
    ]
    
    # Region costs from config
    scaling = 5
    region_costs = scaling * np.random.uniform(0, 1, 12)
    
    # Regions of interest (targets to visit)
    region_of_interest = [8, 9, 10, 11]
    
    # Target positions for visualization (point regions)
    target_positions = []
    for region_idx in region_of_interest:
        region = regions[region_idx]
        x_target = region['xl']  # For point regions, xl=xu
        y_target = region['yl']  # For point regions, yl=yu
        target_positions.append((x_target, y_target))
    
    print(f"Initial state: {x0}")
    print(f"Number of regions: {len(regions)}")
    print(f"Target regions: {region_of_interest}")
    print(f"Planning horizon: {N} steps\n")
    
    # Create parameters
    params = PointMassParams(
        nx=4,
        nu=2,
        N=N,
        dt=dt,
        mass=1.0,
        regions=regions,
        region_control_limits=[{'max_force_x': 1.0, 'max_force_y': 1.0} for _ in range(len(regions))],
        region_costs=region_costs,
        nz=len(regions),
        M=10.0,
        max_velocity=0.8,
    )
    
    # Build STL specification for 4 targets: [8, 9, 10, 11]
    # Sequential constraints: visit 9 before 8, visit 11 before 10
    not_visit8 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[0], neg=True)
    not_visit10 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[2], neg=True)
    
    visit9 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[1], neg=False)
    visit11 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[3], neg=False)

    not_visit8_until_visit9 = not_visit8.until(visit9, 0, params.N)
    not_visit10_until_visit11 = not_visit10.until(visit11, 0, params.N)
    
    # Eventually visit all targets (with 3-step dwell time)
    visit8_eventually = ConvexSetPredicate(name='nf', edge_no=region_of_interest[0], neg=False).always(0, 3).eventually(0, params.N-5)
    visit9_eventually = ConvexSetPredicate(name='nf', edge_no=region_of_interest[1], neg=False).always(0, 3).eventually(0, params.N-5)
    visit10_eventually = ConvexSetPredicate(name='nf', edge_no=region_of_interest[2], neg=False).always(0, 3).eventually(0, params.N-5)
    visit11_eventually = ConvexSetPredicate(name='nf', edge_no=region_of_interest[3], neg=False).always(0, 3).eventually(0, params.N-5)

    # Combined specification
    specification = not_visit8_until_visit9 & not_visit10_until_visit11 & \
                   (visit8_eventually | visit10_eventually) & \
                   visit9_eventually & visit11_eventually
    specification.simplify()
    
    # =====================================================================================
    # Compare LNF vs LT
    # =====================================================================================
    
    for model_type in ['LNF', 'LT']:
        print(f"\n{'='*80}")
        print(colored(f"Running {model_type} model", 'green', attrs=['bold']))
        print(f"{'='*80}\n")
        
        # Create optimizer model
        optimizer_model = gp.Model(f"point_mass_{model_type}")
        
        # Setup point mass dynamics
        point_mass = PointMassDynamics(params)
        point_mass.setup_optimization(optimizer_model, x0)
        
        # Add logic constraints
        if model_type == 'LNF':
            unique_predicate_dict, node_list, edge_list, output = assemble_logic_nf_info(specification)
            z_pi = create_binary_mapping(unique_predicate_dict, point_mass.z_vars)
            _ = logic_nf(optimizer_model, unique_predicate_dict, node_list, edge_list, output, z_pi,
                        add_lt_cuts=False, use_sos1_encoding=False)
        
        elif model_type == 'LT':
            unique_predicate_dict, MT, MU, MV, num_extra_vars = assemble_logic_tree_info(specification)
            
            # Remove zero columns
            non_zero_cols = []
            for j in range(MT.shape[1]):
                if not np.all(MT[:, j] == 0):
                    non_zero_cols.append(j)
            MT = MT[:, non_zero_cols]
            num_extra_vars = MT.shape[1]
            
            z_pi = create_binary_mapping(unique_predicate_dict, point_mass.z_vars)
            _ = logic_tree(optimizer_model, unique_predicate_dict, num_extra_vars, MT, MU, MV, z_pi)
        
        # Solve the model
        print("Solving the problem with temporal logic constraints...")
        optimizer_model.optimize()
        
        # Extract and visualize trajectory
        x_trajectory, u_trajectory = point_mass.solve_and_extract_trajectory()
        
        if x_trajectory is not None:
            print(colored(f"\n{model_type} optimization succeeded!", 'green'))
            print(f"Objective value: {optimizer_model.ObjVal:.4f}")
            print(f"Solve time: {optimizer_model.Runtime:.2f} seconds")
            print(f"Final state: {x_trajectory[-1]}")
            
            save_path = f"trajectory_{model_type}.png"
            point_mass.plot_trajectory(x_trajectory, u_trajectory, target_positions, save_path)
        else:
            print(colored(f"{model_type} optimization failed", 'red'))


if __name__ == '__main__':
    main()
