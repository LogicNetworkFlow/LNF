import numpy as np
from abc import ABC, abstractmethod
from treelib import Tree
from dataclasses import dataclass
import copy
    

class STLFormula:
    """Base class for all STL formulas"""
    _non_predicate_counter = 0  # Class variable for non-predicate nodes in the stltree
    
    @classmethod
    def reset_non_predicate_counter(cls):
        cls._non_predicate_counter = 0

    def __init__(self, neg=False):
        self.neg = neg

    def get_name_no_inst(self) -> str:
        """Get predicate name without instance number"""
        if not self.is_predicate():
            raise ValueError("get_name_no_inst() can only be called on predicates")
        
        if self.is_input:
            return 'input'
        elif self.is_output:
            return 'output'
        return f"{self.name}_t{self.timestep}_ed{self.edge_no}"

    def get_name_with_inst(self) -> str:
        """Get predicate name with instance number"""
        if not self.is_predicate():
            raise ValueError("get_name_with_inst() can only be called on predicates")
        
        if self.is_input or self.is_output:
            return self.get_name_no_inst()
        return f"{self.get_name_no_inst()}_i{self.instance}"

    def negation(self):
        """Toggle negation flag"""
        self.neg = not self.neg
        
    def clone(self):
        """Create a copy of this formula"""
        return STLFormula(self.neg)

    def conjunction(self, other):
        r"""
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which represents the conjunction
        (and) of this formula (:math:`\\varphi`) and another one (:math:`\\varphi_{other}`):

        .. math::

            \\varphi_{new} = \\varphi \land \\varphi_{other}

        :param other:   The :class:`.STLFormula` :math:`\\varphi_{other}`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`

        .. note::

            Conjuction can also be represented with the ``&`` operator, i.e.,
            ::

                c = a & b

            is equivalent to
            ::

                c = a.conjuction(b)

        """
        return STLTree([self.clone(), other.clone()], "and", [0, 0])

    def disjunction(self, other):
        r"""
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which represents the disjunction
        (or) of this formula (:math:`\\varphi`) and another one (:math:`\\varphi_{other}`):

        .. math::

            \\varphi_{new} = \\varphi \lor \\varphi_{other}

        :param other:   The :class:`.STLFormula` :math:`\\varphi_{other}`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`

        .. note::

            Disjunction can also be represented with the ``|`` operator, i.e.,
            ::

                c = a | b

            is equivalent to
            ::

                c = a.disjunction(b)

        """
        return STLTree([self.clone(), other.clone()], "or", [0, 0])

    def __and__(self, other):
        """
        Syntatic sugar so we can write `one_and_two = one & two`
        """
        return self.conjunction(other)

    def __or__(self, other):
        """
        Syntatic sugar so we can write `one_or_two = one | two`
        """
        return self.disjunction(other)

    def always(self, t1, t2):
        r"""
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that this
        formula (:math:`\\varphi`) holds for all of the timesteps between
        :math:`t_1` and :math:`t_2`:

        .. math::

            \\varphi_{new} = G_{[t_1,t_2]}(\\varphi)


        :param t1:  An integer representing the delay :math:`t_1`
        :param t2:  An integer representing the deadline :math:`t_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        time_interval = [t for t in range(t1, t2+1)]
        ls_subformula = [copy.deepcopy(self) for _ in time_interval]
        formula = STLTree(ls_subformula, "and", time_interval)
        
        return formula

    def eventually(self, t1, t2):
        r"""
        Return a new :class:`.STLTree` :math:`\\varphi_{new}` which ensures that this
        formula (:math:`\\varphi`) holds for at least one timestep between
        :math:`t_1` and :math:`t_2`:

        .. math::

            \\varphi_{new} = F_{[t_1,t_2]}(\\varphi)


        :param t1:  An integer representing the delay :math:`t_1`
        :param t2:  An integer representing the deadline :math:`t_2`

        :return: An :class:`.STLTree` representing :math:`\\varphi_{new}`
        """
        time_interval = [t for t in range(t1, t2+1)]
        ls_subformula = [copy.deepcopy(self) for _ in time_interval]
        formula = STLTree(ls_subformula, "or", time_interval)
        
        return formula

    def until(self, other, t1, t2):
        self_until_tprime = []
        
        # B becomes true at some t_prime
        for t_prime in range(t1, t2+1):
            not_other_before = []
            for t in range(t1, t_prime):
                not_other_copy = copy.deepcopy(other)
                not_other_copy.negation()
                not_other_before.append(not_other_copy)
            
            ls_subformula = [copy.deepcopy(self) for _ in range(t1, t_prime)]
            ls_subformula.extend(not_other_before)
            ls_subformula.append(copy.deepcopy(other))
            
            time_interval = [t for t in range(t1, t_prime)]
            time_interval.extend([t for t in range(t1, t_prime)])
            time_interval.append(t_prime)
            
            self_until_tprime.append(STLTree(ls_subformula, "and", time_interval))
        
        # A holds throughout [t1, t2] AND B never holds throughout [t1, t2]
        ls_subformula_never = [copy.deepcopy(self) for _ in range(t1, t2+1)]  # A holds always
        
        for t in range(t1, t2+1):  # B never holds
            not_other_copy = copy.deepcopy(other)
            not_other_copy.negation()
            ls_subformula_never.append(not_other_copy)
        
        time_interval_never = [t for t in range(t1, t2+1)]  # For A
        time_interval_never.extend([t for t in range(t1, t2+1)])  # For not B
        
        self_until_tprime.append(STLTree(ls_subformula_never, "and", time_interval_never))
        
        return STLTree(self_until_tprime, "or", [0] * len(self_until_tprime))

    def is_predicate(self):
        """
        Indicate whether this formula is a predicate.

        :return:    A boolean which is ``True`` only if this is a predicate.
        """
        pass

    def is_state_formula(self):
        """
        Indicate whether this formula is a state formula, e.g.,
        a predicate or the result of boolean operations over
        predicates.

        :return:    A boolean which is ``True`` only if this is a state formula.
        """
        pass

    def is_neg(self):
        """Getter for negation flag"""
        return self.neg

    def is_disjunctive_state_formula(self):
        """
        Indicate whether this formula is a state formula defined by
        only disjunctions (or) over predicates.

        :return:     A boolean which is ``True`` only if this is a disjunctive state formula.
        """
        pass

    def is_conjunctive_state_formula(self):
        """
        Indicate whether this formula is a state formula defined by
        only conjunctions (and) over predicates.

        :return:     A boolean which is ``True`` only if this is a conjunctive state formula.
        """
        pass

class STLTree(STLFormula):
    r"""
    Describes an STL formula :math:`\\varphi` which is made up of
    operations over :class:`.STLFormula` objects. This defines a tree structure,
    so that, for example, the specification

    .. math::

        \\varphi = G_{[0,3]} \\pi_1 \land F_{[0,3]} \\pi_2

    is represented by the tree

    .. graphviz::

        digraph tree {
            root [label="phi"];
            G [label="G"];
            F [label="F"];
            n1 [label="pi_1"];
            n2 [label="pi_1"];
            n3 [label="pi_1"];
            n4 [label="pi_2"];
            n5 [label="pi_2"];
            n6 [label="pi_2"];

            root -> G;
            root -> F;
            G -> n1;
            G -> n2;
            G -> n3;
            F -> n4;
            F -> n5;
            F -> n6;
        }

    where each node is an :class:`.STLFormula` and the leaf nodes are :class:`.LinearPredicate` objects.


    Each :class:`.STLTree` is defined by a list of :class:`.STLFormula` objects
    (the child nodes in the tree) which are combined together using either conjunction or
    disjunction.

    :param ls_subformula:     A list of :class:`.STLFormula` objects (formulas or
                                predicates) that we'll use to construct this formula.
    :param combination_type:    A string representing the type of operation we'll use
                                to combine the child nodes. Must be either ``"and"`` or ``"or"``.
    :param timesteps:           A list of timesteps that the subformulas must hold at.
                                This is needed to define the temporal operators.
    """
    def __init__(self, ls_subformula, combination_type, ls_timestep):
        super().__init__(False)
        assert combination_type in ["and", "or"]
        self.combination_type = combination_type
        self.ls_subformula = ls_subformula
        self.ls_timestep = ls_timestep

        self.label_non_predicate_var = STLFormula._non_predicate_counter
        STLFormula._non_predicate_counter += 1

        self._propagate_timesteps(0)  # Propagate timesteps during tree construction

    def _propagate_timesteps(self, base_timestep):
        """Propagate timesteps to all subformulas during tree construction"""
        for i, subformula in enumerate(self.ls_subformula):
            if subformula.is_predicate():
                subformula.timestep = base_timestep + self.ls_timestep[i]
            elif isinstance(subformula, STLTree):
                subformula._propagate_timesteps(base_timestep + self.ls_timestep[i])

    def clone(self):
        return STLTree([f.clone() for f in self.ls_subformula], self.combination_type, self.ls_timestep.copy())
    
    def is_predicate(self):
        return False

    def negation(self):
        """
        Apply negation to this formula using DeMorgan's laws.
        For predicates: toggles the negation flag.
        For trees: applies DeMorgan's laws and pushes negation down to the leaves.
        """
        if self.is_predicate():
            # For predicates, just toggle the negation flag
            self.neg = not self.neg
        else:
            # For trees, apply DeMorgan's laws without toggling self.neg
            # Flip combination type (DeMorgan's law)
            if self.combination_type == "and":
                self.combination_type = "or"
            else:
                self.combination_type = "and"
            
            # Recursively negate all subformulas
            for subformula in self.ls_subformula:
                subformula.negation()

    def simplify(self):
        """
        Modify this formula to reduce the depth of the formula tree while preserving
        logical equivalence.

        A shallower formula tree can result in a more efficient binary encoding in some
        cases.
        """
        mod = True
        while mod:
            # Just keep trying to flatten until we don't get any improvement
            mod = self.flatten(self)

        # Assign predicate names after tree is finalized
        self.unique_predicate_dict = {}
        self.predicate_list = []
        self._collect_predicate_names()

    def _collect_predicate_names(self, tree=None):
        """Internal method to collect and assign predicate names"""
        # Start with self on first call
        if tree is None:
            tree = self
        
        if tree.is_predicate():
            base_name = tree.get_name_no_inst()
            
            # Get current instance count and increment
            tree.instance = self.unique_predicate_dict.get(base_name, 0)
            self.unique_predicate_dict[base_name] = tree.instance + 1
            
            self.predicate_list.append({'name': tree.get_name_with_inst(), 'in': [], 'out': []})
        else:
            for subtree in tree.ls_subformula:
                self._collect_predicate_names(subtree)

    def flatten(self, formula):
        r"""
        Reduce the depth of the given :class:`STLFormula` by combining adjacent
        layers with the same logical operation. This preserves the meaning of the
        formula, since, for example,

        ..math::

            (a \land b) \land (c \land d) = a \land b \land c \land d

        :param formula: The formula to modify

        :return made_modification: boolean flag indicating whether the formula was changed.l

        """
        made_modification = False

        for subformula in formula.ls_subformula:
            if subformula.is_predicate():
                pass
            else:
                if formula.combination_type == subformula.combination_type:
                    # Remove the subformula
                    i = formula.ls_subformula.index(subformula)
                    formula.ls_subformula.pop(i)
                    st = formula.ls_timestep.pop(i)

                    # Add all the subformula's subformulas instead
                    formula.ls_subformula += subformula.ls_subformula
                    formula.ls_timestep += [t+st for t in subformula.ls_timestep]
                    made_modification = True

                made_modification = self.flatten(subformula) or made_modification

        return made_modification

    def get_all_inequalities(self):
        As = []
        bs = []
        for subformula in self.ls_subformula:
            A, b = subformula.get_all_inequalities()
            As.append(A)
            bs.append(b)
        A = np.vstack(As)
        b = np.hstack(bs)

        return A, b

    def __str__(self):
        """Print tree structure with negations and combination types"""
        return self._str_helper("")

    def _str_helper(self, indent=""):
        """Helper method for recursive string representation"""
        result = indent + ("NOT " if self.neg else "") + self.combination_type + "\n"
        for i, subformula in enumerate(self.ls_subformula):
            if isinstance(subformula, STLTree):
                result += subformula._str_helper(indent + "  ")
            else:  # Predicate
                result += indent + "  " + ("NOT " if subformula.neg else "") + \
                            f"P({subformula.name}, {subformula.edge_no}, t={subformula.timestep})\n"
        return result
