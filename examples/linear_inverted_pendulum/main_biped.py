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
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import List, Dict
from termcolor import colored


@dataclass
class BipedParams:
    """Parameters for bipedal locomotion dynamics and environment"""
    nx: int  # state dimension [x, y, xdot, ydot, theta]
    nu: int  # control dimension [x_foot, y_foot, theta_dot]
    N: int  # horizon length
    dt: float  # time step
    mass: float  # mass of the robot
    g: float  # gravity
    h: float  # height of COM

    # Environment constraints
    regions: List[Dict]  # Region limits
    region_costs: List[float]  # Region-specific costs
    nz: int  # number of regions
    M: float  # big M for spatial constraints
    Mt: float  # big M for trigonometric constraints
    max_velocity: float  # maximum velocity
    left_stand_first: bool  # True if left foot is first standing foot


class BipedDynamics:
    """Bipedal locomotion dynamics using Linear Inverted Pendulum model"""
    
    def __init__(self, params: BipedParams):
        self.p = params
        
        # Calculate pendulum natural frequency
        self.omega = np.sqrt(self.p.g / self.p.h)
        
        # Binary variable counts
        self.nk = 4  # number of pieces for trig approximation
        self.nb = self.p.nz + self.nk  # total binary variables

    def _get_sin_cos_approximation(self, index, num_pieces):
        """Get piecewise linear approximation of sin and cos within [-π, π]"""
        total_range = 2 * np.pi
        piece_width = total_range / num_pieces
        range_start = -np.pi
        
        breakpoints = [range_start + piece_width * i for i in range(num_pieces + 1)]
        sin_values = [np.sin(bp) for bp in breakpoints]
        cos_values = [np.cos(bp) for bp in breakpoints]
        
        # Calculate slopes and intercepts
        sin_slope = (sin_values[index + 1] - sin_values[index]) / (breakpoints[index + 1] - breakpoints[index])
        sin_intercept = sin_values[index + 1] - sin_slope * breakpoints[index + 1]
        
        cos_slope = (cos_values[index + 1] - cos_values[index]) / (breakpoints[index + 1] - breakpoints[index])
        cos_intercept = cos_values[index + 1] - cos_slope * breakpoints[index + 1]
        
        return (sin_slope, sin_intercept, cos_slope, cos_intercept, breakpoints[index], breakpoints[index + 1])

    def setup_optimization(self, model, x0: np.ndarray):
        """Setup optimization problem with bipedal dynamics"""
        self.model = model
        
        # Create variables
        x = {}  # States
        u = {}  # Controls
        z = {}  # Binary variables for regions and trig
        sin_theta = {}  # Sin approximation
        cos_theta = {}  # Cos approximation
        
        # State variables
        for t in range(self.p.N + 1):
            x[t,0] = self.model.addVar(lb=-self.p.M, ub=self.p.M, name=f'x_{t}_0')  # x position
            x[t,1] = self.model.addVar(lb=-self.p.M, ub=self.p.M, name=f'x_{t}_1')  # y position
            x[t,2] = self.model.addVar(lb=-self.p.max_velocity, ub=self.p.max_velocity, name=f'x_{t}_2')  # x velocity
            x[t,3] = self.model.addVar(lb=-self.p.max_velocity, ub=self.p.max_velocity, name=f'x_{t}_3')  # y velocity
            x[t,4] = self.model.addVar(lb=-np.pi, ub=np.pi, name=f'x_{t}_4')  # theta
        
        # Control variables
        for t in range(self.p.N):
            u[t,0] = self.model.addVar(lb=-0.25, ub=0.25, name=f'u_{t}_0')  # foot x
            u[t,1] = self.model.addVar(lb=-0.25, ub=0.25, name=f'u_{t}_1')  # foot y
            u[t,2] = self.model.addVar(lb=-np.pi/15, ub=np.pi/15, name=f'u_{t}_2')  # theta_dot
        
        # Binary variables
        for t in range(self.p.N + 1):
            for zz in range(self.nb):
                z[t,zz] = self.model.addVar(vtype=GRB.BINARY, name=f'z_{t}_{zz}')
        
        # Trigonometric approximation variables
        for t in range(self.p.N + 1):
            sin_theta[t] = self.model.addVar(lb=-1, ub=1, name=f'sin_theta_{t}')
            cos_theta[t] = self.model.addVar(lb=-1, ub=1, name=f'cos_theta_{t}')
        
        # Initial state constraints
        for i in range(self.p.nx):
            self.model.addConstr(x[0,i] == x0[i])
        
        # Setup state transition matrices (Linear Inverted Pendulum)
        g = self.p.g
        leg_length = self.p.h
        omega = np.sqrt(g / leg_length)
        T = self.p.dt
        
        E = np.eye(self.p.nx)
        F = np.zeros((self.p.nx, self.p.nu))
        
        # LIP dynamics for x, y, xdot, ydot
        E_ = np.zeros((4, 4))
        E_[:2, :2] = np.eye(2)
        E_[:2, 2:] = np.eye(2) * np.sinh(omega*T) / omega
        E_[2:, 2:] = np.eye(2) * np.cosh(omega*T)
        E[:4, :4] = E_
        
        F_ = np.zeros((4, 2))
        F_[:2, :2] = np.eye(2) * (1 - np.cosh(omega*T))
        F_[2:, :2] = np.eye(2) * (-omega * np.sinh(omega*T))
        F[:4, :2] = F_
        F[4, 2] = self.p.dt  # theta dynamics
        
        # Dynamics constraints
        for t in range(self.p.N):
            for i in range(self.p.nx):
                expr = gp.LinExpr()
                for j in range(self.p.nx):
                    expr.add(x[t,j], E[i,j])
                for j in range(self.p.nu):
                    expr.add(u[t,j], F[i,j])
                self.model.addConstr(x[t+1,i] == expr)
        
        # Region constraints using Big-M
        for t in range(self.p.N):
            # Exactly one region at a time
            self.model.addConstr(sum(z[t,r] for r in range(self.p.nz)) == 1)
            
            # Foot position
            foot_x = x[t,0] + u[t,0]
            foot_y = x[t,1] + u[t,1]
            
            # Region bounds for foot placement
            for r, region in enumerate(self.p.regions):
                self.model.addConstr(foot_x <= region['xu'] + self.p.M * (1 - z[t,r]))
                self.model.addConstr(foot_x >= region['xl'] - self.p.M * (1 - z[t,r]))
                self.model.addConstr(foot_y <= region['yu'] + self.p.M * (1 - z[t,r]))
                self.model.addConstr(foot_y >= region['yl'] - self.p.M * (1 - z[t,r]))
        
        # Trigonometric approximation constraints
        for t in range(self.p.N + 1):
            angle_start_idx = self.p.nz
            
            # Exactly one angle piece selected
            self.model.addConstr(sum(z[t, angle_start_idx + i] for i in range(self.nk)) == 1)
            
            # Piecewise linear constraints
            for i in range(self.nk):
                sin_slope, sin_int, cos_slope, cos_int, theta_lb, theta_ub = self._get_sin_cos_approximation(i, self.nk)
                piece_var = z[t, angle_start_idx + i]
                
                # Theta bounds
                self.model.addConstr(x[t,4] <= theta_ub + (1 - piece_var) * self.p.Mt)
                self.model.addConstr(x[t,4] >= theta_lb - (1 - piece_var) * self.p.Mt)
                
                # Sin/Cos approximations
                self.model.addConstr(sin_theta[t] <= sin_slope * x[t,4] + sin_int + (1 - piece_var) * self.p.Mt)
                self.model.addConstr(sin_theta[t] >= sin_slope * x[t,4] + sin_int - (1 - piece_var) * self.p.Mt)
                self.model.addConstr(cos_theta[t] <= cos_slope * x[t,4] + cos_int + (1 - piece_var) * self.p.Mt)
                self.model.addConstr(cos_theta[t] >= cos_slope * x[t,4] + cos_int - (1 - piece_var) * self.p.Mt)
        
        # Foothold bounds (alternating left/right foot)
        for t in range(self.p.N):
            if t % 2 == (0 if self.p.left_stand_first else 1):
                fh_ub = np.array([0, 1])
                fh_lb = np.array([0, -2.5])
            else:
                fh_ub = np.array([0, -1])
                fh_lb = np.array([0, 2.5])
            
            # Transform bounds to global frame
            expr1_ub = cos_theta[t] * fh_ub[0] - sin_theta[t] * fh_ub[1]
            expr2_ub = sin_theta[t] * fh_ub[0] + cos_theta[t] * fh_ub[1]
            self.model.addQConstr(
                (u[t,0] - expr1_ub) * (u[t,0] - expr1_ub) + 
                (u[t,1] - expr2_ub) * (u[t,1] - expr2_ub) <= 0.9*0.9,
                name=f"footbd_{t}_1"
            )
            
            expr1_lb = cos_theta[t] * fh_lb[0] - sin_theta[t] * fh_lb[1]
            expr2_lb = sin_theta[t] * fh_lb[0] + cos_theta[t] * fh_lb[1]
            self.model.addQConstr(
                (u[t,0] - expr1_lb) * (u[t,0] - expr1_lb) + 
                (u[t,1] - expr2_lb) * (u[t,1] - expr2_lb) <= 3.1*3.1,
                name=f"footbd_{t}_2"
            )
            
            # Maintain foothold constraints for next steps
            if t < self.p.N - 1:
                u_x_next = (x[t,0] + u[t,0]) - x[t+1,0]
                u_y_next = (x[t,1] + u[t,1]) - x[t+1,1]
                
                expr1_ub_next = cos_theta[t+1] * fh_ub[0] - sin_theta[t+1] * fh_ub[1]
                expr2_ub_next = sin_theta[t+1] * fh_ub[0] + cos_theta[t+1] * fh_ub[1]
                self.model.addQConstr(
                    (u_x_next - expr1_ub_next) * (u_x_next - expr1_ub_next) +
                    (u_y_next - expr2_ub_next) * (u_y_next - expr2_ub_next) <= 0.9*0.9,
                    name=f"footbd_maintain_{t}_1"
                )
                
                expr1_lb_next = cos_theta[t+1] * fh_lb[0] - sin_theta[t+1] * fh_lb[1]
                expr2_lb_next = sin_theta[t+1] * fh_lb[0] + cos_theta[t+1] * fh_lb[1]
                self.model.addQConstr(
                    (u_x_next - expr1_lb_next) * (u_x_next - expr1_lb_next) +
                    (u_y_next - expr2_lb_next) * (u_y_next - expr2_lb_next) <= 3.1*3.1,
                    name=f"footbd_maintain_{t}_2"
                )
            
            if t < self.p.N - 2:
                u_x_next_next = (x[t,0] + u[t,0]) - x[t+2,0]
                u_y_next_next = (x[t,1] + u[t,1]) - x[t+2,1]
                
                expr1_ub_next_next = cos_theta[t+2] * fh_ub[0] - sin_theta[t+2] * fh_ub[1]
                expr2_ub_next_next = sin_theta[t+2] * fh_ub[0] + cos_theta[t+2] * fh_ub[1]
                self.model.addQConstr(
                    (u_x_next_next - expr1_ub_next_next) * (u_x_next_next - expr1_ub_next_next) +
                    (u_y_next_next - expr2_ub_next_next) * (u_y_next_next - expr2_ub_next_next) <= 0.9*0.9,
                    name=f"footbd_maintain2_{t}_1"
                )
                
                expr1_lb_next_next = cos_theta[t+2] * fh_lb[0] - sin_theta[t+2] * fh_lb[1]
                expr2_lb_next_next = sin_theta[t+2] * fh_lb[0] + cos_theta[t+2] * fh_lb[1]
                self.model.addQConstr(
                    (u_x_next_next - expr1_lb_next_next) * (u_x_next_next - expr1_lb_next_next) +
                    (u_y_next_next - expr2_lb_next_next) * (u_y_next_next - expr2_lb_next_next) <= 3.1*3.1,
                    name=f"footbd_maintain2_{t}_2"
                )
        
        # Objective function
        obj = gp.QuadExpr()
        
        # Footstep lateral distance penalty
        for t in range(self.p.N):
            if t % 2 == (0 if self.p.left_stand_first else 1):
                preferred_fh = np.array([0, 0.15])
            else:
                preferred_fh = np.array([0, -0.15])
            
            expr1_pref = cos_theta[t] * preferred_fh[0] - sin_theta[t] * preferred_fh[1]
            expr2_pref = sin_theta[t] * preferred_fh[0] + cos_theta[t] * preferred_fh[1]
            obj.add(10 * (u[t,0] - expr1_pref) * (u[t,0] - expr1_pref))
            obj.add(10 * (u[t,1] - expr2_pref) * (u[t,1] - expr2_pref))
        
        # Region costs
        for t in range(self.p.N + 1):
            for r in range(self.p.nz):
                obj.add(10 * self.p.region_costs[r] * z[t,r])
        
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        
        # Store variables
        self.x_vars = x
        self.u_vars = u
        self.z_vars = z
        self.sin_vars = sin_theta
        self.cos_vars = cos_theta

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
                       target_regions=None, save_path=None):
        """Plot trajectory with environment"""
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_facecolor("gray")
        
        # Plot regions
        for i, region in enumerate(self.p.regions):
            rectangle = plt.Rectangle(
                (region['xl'], region['yl']), 
                (region['xu'] - region['xl']), 
                (region['yu'] - region['yl']), 
                facecolor=region['color'],
                edgecolor='brown',
                linewidth=0.5,
                zorder=2
            )
            ax.add_patch(rectangle)
            
            # Add region labels
            region_center_x = (region['xl'] + region['xu']) / 2
            region_center_y = (region['yl'] + region['yu']) / 2
            cost = self.p.region_costs[i]
            plt.text(
                region_center_x, region_center_y,
                f"R{i}\n{cost:.2f}",
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=7,
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2')
            )
        
        # Mark target regions with stars
        if target_regions is not None:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for pair_idx in range(len(target_regions) // 2):
                pair_number = pair_idx + 1
                circle_color = colors[pair_idx % len(colors)]
                
                region1_idx = target_regions[pair_idx * 2]
                region2_idx = target_regions[pair_idx * 2 + 1]
                
                for region_idx in [region1_idx, region2_idx]:
                    region = self.p.regions[region_idx]
                    center_x = (region['xl'] + region['xu']) / 2
                    center_y = (region['yl'] + region['yu']) / 2
                    
                    circle = plt.Circle((center_x, center_y), 0.075, 
                                      facecolor=circle_color, edgecolor='black', 
                                      linewidth=2, zorder=9)
                    ax.add_patch(circle)
                    
                    plt.text(center_x, center_y, str(pair_number), 
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=10, color='white', weight='bold', zorder=10)
        
        # Plot CoM trajectory
        for i in range(len(x_traj)-1):
            if i % 2 == 0:
                ax.scatter(x_traj[i, 0], x_traj[i, 1], s=3, zorder=3, color="grey")
            else:
                ax.scatter(x_traj[i, 0], x_traj[i, 1], s=3, zorder=3, color="lavender")
        
        # Plot heading angles
        ang = x_traj[:-1,4]
        ax.plot([x_traj[:-1, 0], x_traj[:-1, 0]+0.15*np.cos(ang)], 
               [x_traj[:-1, 1], x_traj[:-1, 1]+0.15*np.sin(ang)], 
               color='red', alpha=0.5, linewidth=0.3)
        
        # Plot footsteps
        ax.plot(x_traj[:-1, 0] + u_traj[:, 0], x_traj[:-1, 1] + u_traj[:, 1], 
               linestyle='--', color='blue', alpha=0.8, linewidth=0.5)
        
        for i in range(len(u_traj)):
            if i % 2 == 0:
                ax.scatter(x_traj[i, 0] + u_traj[i, 0], x_traj[i, 1] + u_traj[i, 1], 
                         marker='*', color="grey", s=3, zorder=3)
            else:
                ax.scatter(x_traj[i, 0] + u_traj[i, 0], x_traj[i, 1] + u_traj[i, 1], 
                         marker='*', color="lavender", s=3, zorder=3)
        
        # Mark start
        plt.scatter(x_traj[0, 0], x_traj[0, 1], color='blue', s=150, marker='o', 
                   label='Start', zorder=5, edgecolor='white', linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Bipedal Robot Locomotion')
        plt.legend(loc='upper right')
        
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
    print(colored("Bipedal Locomotion with Temporal Logic Constraints", 'cyan', attrs=['bold']))
    print("="*80 + "\n")
    
    # Problem parameters
    N = 25
    dt = 0.5
    x0 = np.array([1.40, 1.60, 0.0, 0.0, 0.0])  # [x, y, xdot, ydot, theta]
    
    # Define regions (20 regions from config)
    regions = [
        {'xl': 0.096, 'xu': 0.596, 'yl': 1.741, 'yu': 2.241, 'color': 'lightcoral'},
        {'xl': 0.360, 'xu': 0.860, 'yl': 1.156, 'yu': 1.656, 'color': 'orange'},
        {'xl': 1.679, 'xu': 2.179, 'yl': 1.982, 'yu': 2.482, 'color': 'lime'},
        {'xl': 1.133, 'xu': 1.633, 'yl': 1.246, 'yu': 1.746, 'color': 'yellow'},
        {'xl': 0.931, 'xu': 1.431, 'yl': 2.131, 'yu': 2.631, 'color': 'lightcyan'},
        {'xl': 1.085, 'xu': 1.585, 'yl': 0.440, 'yu': 0.940, 'color': 'greenyellow'},
        {'xl': 2.127, 'xu': 2.627, 'yl': 2.051, 'yu': 2.551, 'color': 'pink'},
        {'xl': 1.931, 'xu': 2.431, 'yl': 1.493, 'yu': 1.993, 'color': 'lavender'},
        {'xl': 0.153, 'xu': 0.653, 'yl': 2.355, 'yu': 2.855, 'color': 'lightblue'},
        {'xl': 2.098, 'xu': 2.598, 'yl': 0.740, 'yu': 1.240, 'color': 'lightsalmon'},
        {'xl': 1.698, 'xu': 2.198, 'yl': 0.235, 'yu': 0.735, 'color': 'thistle'},
        {'xl': 1.479, 'xu': 1.979, 'yl': 0.920, 'yu': 1.420, 'color': 'wheat'},
        {'xl': 2.423, 'xu': 2.923, 'yl': 1.380, 'yu': 1.880, 'color': 'palegreen'},
        {'xl': 0.209, 'xu': 0.709, 'yl': 0.318, 'yu': 0.818, 'color': 'powderblue'},
        {'xl': 2.442, 'xu': 2.942, 'yl': 0.311, 'yu': 0.811, 'color': 'peachpuff'},
        {'xl': 1.836, 'xu': 2.336, 'yl': 2.473, 'yu': 2.973, 'color': 'lightgrey'},
        {'xl': 1.012, 'xu': 1.512, 'yl': 1.713, 'yu': 2.213, 'color': 'lightsteelblue'},
        {'xl': 0.818, 'xu': 1.318, 'yl': 0.867, 'yu': 1.367, 'color': 'lightseagreen'},
        {'xl': 0.653, 'xu': 1.153, 'yl': 0.335, 'yu': 0.835, 'color': 'lightyellow'},
        {'xl': 0.489, 'xu': 0.989, 'yl': 2.066, 'yu': 2.566, 'color': 'lightpink'},
    ]
    
    # Region costs
    region_costs = np.random.uniform(0, 1, len(regions))
    
    # Target regions (pairs: each pair can visit either region)
    region_of_interest = [0, 1, 2, 6, 8, 11, 12, 13, 14, 18]
    
    print(f"Initial state: {x0}")
    print(f"Number of regions: {len(regions)}")
    print(f"Target region pairs: {[region_of_interest[i:i+2] for i in range(0, len(region_of_interest), 2)]}")
    print(f"Planning horizon: {N} steps\n")
    
    # Create parameters
    params = BipedParams(
        nx=5,
        nu=3,
        N=N,
        dt=dt,
        mass=1.0,
        g=9.81,
        h=1.0,
        regions=regions,
        region_costs=region_costs,
        nz=len(regions),
        M=4.0,
        Mt=7.0,
        max_velocity=1.0,
        left_stand_first=True
    )
    
    # Build STL specification
    # 5 pairs of targets: visit at least one from each pair
    visit1 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[0], neg=False).always(0, 1).eventually(0, N-2)
    visit2 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[1], neg=False).always(0, 1).eventually(0, N-2)
    visit3 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[2], neg=False).always(0, 1).eventually(0, N-2)
    visit4 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[3], neg=False).always(0, 1).eventually(0, N-2)
    visit5 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[4], neg=False).always(0, 1).eventually(0, N-2)
    visit6 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[5], neg=False).always(0, 1).eventually(0, N-2)
    visit7 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[6], neg=False).always(0, 1).eventually(0, N-2)
    visit8 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[7], neg=False).always(0, 1).eventually(0, N-2)
    visit9 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[8], neg=False).always(0, 1).eventually(0, N-2)
    visit10 = ConvexSetPredicate(name='nf', edge_no=region_of_interest[9], neg=False).always(0, 1).eventually(0, N-2)
    
    # Combined specification: visit one from each pair
    specification = (visit1 | visit2) & (visit3 | visit4) & (visit5 | visit6) & (visit7 | visit8) & (visit9 | visit10)
    specification.simplify()
    
    # =====================================================================================
    # Compare LNF vs LT
    # =====================================================================================
    
    for model_type in ['LNF', 'LT']:
        print(f"\n{'='*80}")
        print(colored(f"Running {model_type} model", 'green', attrs=['bold']))
        print(f"{'='*80}\n")
        
        # Create optimizer model
        optimizer_model = gp.Model(f"biped_{model_type}")
        
        # Setup bipedal dynamics
        biped = BipedDynamics(params)
        biped.setup_optimization(optimizer_model, x0)
        
        # Add logic constraints
        if model_type == 'LNF':
            unique_predicate_dict, node_list, edge_list, output = assemble_logic_nf_info(specification)
            z_pi = create_binary_mapping(unique_predicate_dict, biped.z_vars)
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
            
            z_pi = create_binary_mapping(unique_predicate_dict, biped.z_vars)
            _ = logic_tree(optimizer_model, unique_predicate_dict, num_extra_vars, MT, MU, MV, z_pi)
        
        # Solve the model
        print("Solving the problem with temporal logic constraints...")
        optimizer_model.optimize()
        
        # Extract and visualize trajectory
        x_trajectory, u_trajectory = biped.solve_and_extract_trajectory()
        
        if x_trajectory is not None:
            print(colored(f"\n{model_type} optimization succeeded!", 'green'))
            print(f"Objective value: {optimizer_model.ObjVal:.4f}")
            print(f"Solve time: {optimizer_model.Runtime:.2f} seconds")
            print(f"Final state: {x_trajectory[-1]}")
            
            save_path = f"trajectory_biped_{model_type}.png"
            biped.plot_trajectory(x_trajectory, u_trajectory, region_of_interest, save_path)
        else:
            print(colored(f"{model_type} optimization failed", 'red'))


if __name__ == '__main__':
    main()
