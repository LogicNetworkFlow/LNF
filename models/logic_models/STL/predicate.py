import numpy as np
from .formula import STLFormula

class ConvexSetPredicate(STLFormula):
    def __init__(self, name, edge_no, neg=False):
        super().__init__(neg)
        self.name = name
        self.edge_no = edge_no
        self.timestep = 0  # A predicate has a time step starting from 0.
        self.input = []

        # Instance tracking
        self.instance = 0
        self.is_input = False
        self.is_output = False
        self.is_extra = False

    def clone(self):
        return ConvexSetPredicate(self.name, self.edge_no, self.neg)
    
    def is_predicate(self):
        return True

    def is_state_formula(self):
        return True

    def is_disjunctive_state_formula(self):
        pass

    def is_conjunctive_state_formula(self):
        pass
