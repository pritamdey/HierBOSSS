# Required imports

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from scipy.special import gammaln
import copy
import time
from tqdm.auto import tqdm
import networkx as nx
from dataclasses import dataclass
from typing import Optional


add = {"name": "add", "type": 2}
mul = {"name": "mul", "type": 2}
neg = {"name": "neg", "type": 1}
inv = {"name": "inv", "type": 1}
sin = {"name": "sin", "type": 1}
cos = {"name": "cos", "type": 1}
exp = {"name": "exp", "type": 1}
sq = {"name": "sq", "type": 1}
cu = {"name": "cu", "type": 1}

opset = [add, mul, neg, inv, sin, cos, exp, sq, cu]

def split_prob_func(depth, maxdepth, alpha = 0.95, delta = 1.2):
## depth-dependent split probability
    if depth == maxdepth:
        return 0
    else:
        return alpha * ((1 + depth) ** (-delta))
    
def lchild_address(address):
## left child node address
    return 2*address + 1

def rchild_address(address):
## right child node address
    return 2*address + 2

def parent_address(address):
## parent node address
    return math.ceil(address/2) - 1

# Helper for running parallel MCMC chains
def _parallel_chain_worker(args):
    (
        X,
        y,
        K,
        maxdepth,
        prior_params,
        add_intercept,
        wts_init,
        wts_prop,
        opset,
        ftset,
        seed,
        move_weights,
        maxiter,
        burnin,
        thin,
        chain_id,
        progress_queue,
        report_every
    ) = args

    model = HierBOSSS_MCMC(
        X=X,
        y=y,
        K=K,
        maxdepth=maxdepth,
        prior_params=prior_params,
        add_intercept=add_intercept,
        wts_init=wts_init,
        wts_prop=wts_prop,
        opset=opset,
        ftset=ftset
    )

    return model.run_MCMC(
        seed=seed,
        move_weights=move_weights,
        maxiter=maxiter,
        burnin=burnin,
        thin=thin,
        show_progress=False,
        progress_queue=progress_queue,
        chain_id=chain_id,
        report_every=report_every
    )

# Node and Forest Class
@dataclass
class Node:
    address: int = 0
    depth: int = 0
    nchild: int = 0
    op: Optional[dict] = None
    ft: Optional[str] = None

class Forest:
    def __init__(self, ntrees, maxdepth, opset, ftset, alpha=0.95, delta=1.2, wts_init = None):
        # Setting global forest parameters
        self.ntrees = ntrees
        self.maxdepth = maxdepth
        self.max_nodes_per_tree = (2 ** (maxdepth + 1)) - 1
        self.ops = opset
        self.features = ftset
        self.alpha = alpha
        self.delta = delta
        
        # Initializing trees
        if not(wts_init):
            wts_init = [[1 for _ in range(len(opset))], [1 for _ in range(len(ftset))]]
        self.trees = [[None] * self.max_nodes_per_tree for _ in range(ntrees)]
        for i in range(ntrees):
            self.grow_node(wts_init, 0, i)

    # Move Helper 1
    def _clear_subtree(self, node_address, tree_id):
        node = self.trees[tree_id][node_address]
        if node is None:
            return
        
        if node.nchild >= 1:
            left = lchild_address(node_address)
            if left < self.max_nodes_per_tree:
                self._clear_subtree(left, tree_id)
                
        if node.nchild == 2:
            right = rchild_address(node_address)
            if right < self.max_nodes_per_tree:
                self._clear_subtree(right, tree_id)
        self.trees[tree_id][node_address] = None

    # Move Helper 2
    def _get_subtree_copy(self, node_address, tree_id):
        node = self.trees[tree_id][node_address]
        
        out = {
            "node": copy.deepcopy(node),
            "left": None,
            "right": None
        }
        
        if node.nchild >= 1:
            left_address = lchild_address(node_address)
            if left_address < self.max_nodes_per_tree and self.trees[tree_id][left_address] is not None:
                out["left"] = self._get_subtree_copy(left_address, tree_id)
            
        if node.nchild == 2:
            right_address = rchild_address(node_address)
            if right_address < self.max_nodes_per_tree and self.trees[tree_id][right_address] is not None:
                out["right"] = self._get_subtree_copy(right_address, tree_id)

        return out
    
    # Move Helper 3
    def _write_subtree(self, node_address, tree_id, subtree, depth):
        node = copy.deepcopy(subtree["node"])
        node.address = node_address
        node.depth = depth
        self.trees[tree_id][node_address] = node
        
        left_address = lchild_address(node_address)
        right_address = rchild_address(node_address)
        
        if node.nchild >= 1 and subtree["left"] is not None:
            self._write_subtree(left_address, tree_id, subtree["left"], depth+1)
        else:
            if left_address < self.max_nodes_per_tree:
                self._clear_subtree(left_address, tree_id)
                
        if node.nchild == 2 and subtree["right"] is not None:
            self._write_subtree(right_address, tree_id, subtree["right"], depth+1)
        else:
            if right_address < self.max_nodes_per_tree:
                self._clear_subtree(right_address, tree_id)
    
    # Move Helper 4
    def _subtree_height(self, node_address, tree_id):
        node = self.trees[tree_id][node_address]
        if node is None:
            return -1   # empty subtree

        if node.nchild == 0:
            return 0

        left_h = self._subtree_height(lchild_address(node_address), tree_id)

        if node.nchild == 1:
            return 1 + left_h

        right_h = self._subtree_height(rchild_address(node_address), tree_id)
        return 1 + max(left_h, right_h)

    # Move 1: Grow
    def grow_node(self, wts, node_address, tree_id):
        op_wts, ft_wts = wts

        if self.trees[tree_id][node_address] is None:
            depth = 0 if node_address == 0 else self.trees[tree_id][parent_address(node_address)].depth + 1
            self.trees[tree_id][node_address] = Node(address=node_address, depth=depth)

        curdepth = self.trees[tree_id][node_address].depth

        if curdepth < self.maxdepth:
            split_prob = split_prob_func(curdepth, self.maxdepth, alpha=self.alpha, delta=self.delta)
            tosplit = random.choices([True, False], weights=[split_prob, 1 - split_prob], k=1)[0]
        else:
            tosplit = False

        if tosplit:
            chosen_op = random.choices(self.ops, weights=op_wts, k=1)[0]
            self.trees[tree_id][node_address].op = chosen_op
            self.trees[tree_id][node_address].ft = None
            self.trees[tree_id][node_address].nchild = chosen_op["type"]

            left_address = lchild_address(node_address)
            self.trees[tree_id][left_address] = Node(address=left_address, depth=curdepth + 1)
            self.grow_node(wts, left_address, tree_id)

            if chosen_op["type"] == 2:
                right_address = rchild_address(node_address)
                self.trees[tree_id][right_address] = Node(address=right_address, depth=curdepth + 1)
                self.grow_node(wts, right_address, tree_id)
        else:
            self.trees[tree_id][node_address].op = None
            self.trees[tree_id][node_address].ft = random.choices(self.features, weights=ft_wts, k=1)[0]
            self.trees[tree_id][node_address].nchild = 0

    # Move 2: Prune
    def prune_node(self, wts, node_address, tree_id):
        op_wts, ft_wts = wts
        node = self.trees[tree_id][node_address]
        curdepth = node.depth
        
        if node.nchild >= 1:
            left = lchild_address(node_address)
            if left < self.max_nodes_per_tree:
                self._clear_subtree(left, tree_id)
                
        if node.nchild == 2:
            right = rchild_address(node_address)
            if right < self.max_nodes_per_tree:
                self._clear_subtree(right, tree_id)
                
        self.trees[tree_id][node_address] = Node(
            address=node_address,
            depth=curdepth,
            nchild=0,
            op=None,
            ft=random.choices(self.features, weights=ft_wts, k=1)[0]
        )
    
    # Move 3: Change feature
    def exchange_features(self, wts, node_address, tree_id):
        op_wts, ft_wts = wts
        ft_wts_new = copy.deepcopy(ft_wts)
        cur_ft_idx = self.features.index(self.trees[tree_id][node_address].ft)
        ft_wts_new[cur_ft_idx] = 0
        self.trees[tree_id][node_address].ft = random.choices(self.features, weights=ft_wts_new, k=1)[0]
    
    #  Move 4: Change operator
    def exchange_operators(self, wts, node_address, tree_id):
        op_wts, ft_wts = wts
        op_wts_new = copy.deepcopy(op_wts)
        cur_op = self.trees[tree_id][node_address].op
        cur_op_arity = cur_op["type"]
        
        for i, op in enumerate(self.ops):
            if op["type"] != cur_op_arity:
                op_wts_new[i] = 0
        
        cur_op_idx = self.ops.index(cur_op)
        op_wts_new[cur_op_idx] = 0
        
        if sum(op_wts_new) == 0:
            raise ValueError("No same-arity alternative operator available.")
    
        self.trees[tree_id][node_address].op = random.choices(self.ops, weights=op_wts_new, k=1)[0]
    
    #  Move 5: Subtree replacement
    def subtree_replace(self, wts, node_address, tree_id):
        self._clear_subtree(node_address=node_address, tree_id=tree_id)
        self.grow_node(wts=wts, node_address=node_address, tree_id=tree_id)
     
    #  Move 6: Delete node
    def delete_node(self, node_address, tree_id):
        node = self.trees[tree_id][node_address]
        
        curdepth = node.depth
        
        left_address = lchild_address(node_address)
        right_address = rchild_address(node_address)
        
        if node.nchild == 1:
            promote_address = left_address
        
        elif node.nchild == 2:
            children = [left_address, right_address]
            promote_address = random.choice(children)
        
        promoted_subtree = self._get_subtree_copy(promote_address, tree_id)
        
        self._clear_subtree(node_address, tree_id)
        
        self._write_subtree(node_address, tree_id, promoted_subtree, curdepth)
    
    # Move 7: Insert Node
    def insert_node(self, wts, node_address, tree_id):
        op_wts, ft_wts = wts
        old_node = self.trees[tree_id][node_address]
        old_depth = old_node.depth
        
        # check if shifting subtree down by 1 violates maxdepth
        h = self._subtree_height(node_address, tree_id)
        if old_depth + 1 + h > self.maxdepth:
            raise ValueError("Insertion would create a subtree deeper than maxdepth")
        
        old_subtree = self._get_subtree_copy(node_address, tree_id)
        self._clear_subtree(node_address, tree_id)
        
        chosen_op = random.choices(self.ops, weights=op_wts, k=1)[0]
        new_arity = chosen_op["type"]
        
        self.trees[tree_id][node_address] = Node(
            address=node_address,
            depth=old_depth,
            nchild=new_arity,
            op=chosen_op,
            ft=None
        )
        
        left_address = lchild_address(node_address)
        
        self._write_subtree(
            node_address=left_address,
            tree_id=tree_id,
            subtree=old_subtree,
            depth=old_depth+1
        )
        
        if new_arity == 2:
            right_address = rchild_address(node_address)
            
            self.trees[tree_id][right_address] = Node(
                address=right_address,
                depth=old_depth+1,
                nchild=0,
                op=None,
                ft=None
            )
            self.grow_node(wts, right_address, tree_id)
    
    # Getting Tree Statistics
    def tree_statistics(self, tree_id):
        terminal_addresses = []
        nonterminal_addresses = []
        addresses_by_depth = {}
        terminal_count_by_depth = {}
        nonterminal_count_by_depth = {}
        operator_counts = {op["name"]: 0 for op in self.ops}
        feature_counts = {ft: 0 for ft in self.features}
        
        active_addresses = []
        
        # Getting the total terminal and nonterminal node addresses, along with operator and feature counts
        def traverse(node_address):
            node = self.trees[tree_id][node_address]
            if node is None:
                return
            
            active_addresses.append(node_address)
            
            d = node.depth
            addresses_by_depth.setdefault(d, []).append(node_address)
            terminal_count_by_depth.setdefault(d, 0)
            nonterminal_count_by_depth.setdefault(d, 0)
            
            if node.nchild == 0:
                terminal_addresses.append(node_address)
                terminal_count_by_depth[d] += 1
                if node.ft is not None:
                    feature_counts[node.ft] += 1
            
            else:
                nonterminal_addresses.append(node_address)
                nonterminal_count_by_depth[d] += 1
                if node.op is not None:
                    operator_counts[node.op["name"]] += 1
                
                left_address = lchild_address(node_address)
                if left_address < self.max_nodes_per_tree:
                    traverse(left_address)
                
                if node.nchild == 2:
                    right_address = rchild_address(node_address)
                    if right_address < self.max_nodes_per_tree:
                        traverse(right_address)
                        
        traverse(0)
        
        #----------------------
        # Valid move addresses
        #----------------------
        
        # Move 1: grow validity
        valid_grow = [
            addr for addr in terminal_addresses
            if self.trees[tree_id][addr].depth < self.maxdepth
        ]
        
        # Move 2: prune validity
        valid_prune = nonterminal_addresses.copy()
        
        # Move 3: change features validity
        valid_change_feature = []
        if len(self.features) >= 2:
            valid_change_feature = terminal_addresses.copy()
        
        # Move 4: change operators validity
        valid_change_operator = []
        for addr in nonterminal_addresses:
            node = self.trees[tree_id][addr]
            cur_op = node.op
            cur_arity = cur_op["type"]
            
            same_arity_ops = [op for op in self.ops if op["type"] == cur_arity]
            # need at least one alternative operator
            if len(same_arity_ops) >= 2:
                valid_change_operator.append(addr)
        
        # Move 5: subtree replace validity
        valid_subtree_replace = active_addresses.copy()
        
        # Move 6: delete node validity
        valid_delete_node = nonterminal_addresses.copy()
        
        # Move 7: insert node validity
        valid_insert_node = []
        for addr in active_addresses:
            node = self.trees[tree_id][addr]
            h = self._subtree_height(addr, tree_id)
            
            # old subtree will be shifted down by 1 level
            if node.depth + 1 + h <= self.maxdepth:
                valid_insert_node.append(addr)
        
        return{
        "terminal_addresses": sorted(terminal_addresses),
        "nonterminal_addresses": sorted(nonterminal_addresses),
        "addresses_by_depth": {k: sorted(v) for k, v in sorted(addresses_by_depth.items())},
        "terminal_count_by_depth": dict(sorted(terminal_count_by_depth.items())),
        "nonterminal_count_by_depth": dict(sorted(nonterminal_count_by_depth.items())),
        "operator_counts": operator_counts,
        "feature_counts": feature_counts,
        "valid_move_addresses": {
            "grow": sorted(valid_grow),
            "prune": sorted(valid_prune),
            "change_feature": sorted(valid_change_feature),
            "change_operator": sorted(valid_change_operator),
            "subtree_replace": sorted(valid_subtree_replace),
            "delete_node": sorted(valid_delete_node),
            "insert_node": sorted(valid_insert_node),
        }
        }

    # Helper: Individial tree evaluation at each observation
    def _eval_node(self, node_address, tree_id, X):
        node = self.trees[tree_id][node_address]

        if node.nchild == 0:
            ft_name = node.ft
            if ft_name is None:
                raise ValueError("Terminal node has no feature.")
            ft_idx = self.features.index(ft_name)
            return X[:, ft_idx]
        
        op_name = node.op["name"]

        left_address = lchild_address(node_address)
        left_val = self._eval_node(left_address, tree_id, X)

        if node.nchild == 1:
            if op_name == "neg":
                return -left_val
            elif op_name == "inv":
                out = np.array(left_val, dtype=float).copy()
                eps = 1e-8
                out[np.abs(out) < eps] = np.sign(out[np.abs(out) < eps] + eps) * eps
                return 1.0 / out
            elif op_name == "sin":
                return np.sin(left_val)
            elif op_name == "cos":
                return np.cos(left_val)
            elif op_name == "exp":
                return np.exp(np.clip(left_val, -20, 20))
            elif op_name == "sq":
                out = np.array(left_val, dtype=float).copy()
                # out = np.clip(out, -50, 50)
                return out ** 2
            elif op_name == "cu":
                out = np.array(left_val, dtype=float).copy()
                # out = np.clip(out, -20, 20)
                return out ** 3 

        elif node.nchild == 2:
            right_address = rchild_address(node_address)
            right_val = self._eval_node(right_address, tree_id, X)

            if op_name == "add":
                return left_val + right_val
            elif op_name == "mul":
                return left_val * right_val

    # Get full symbolic expression for one tree  as  a string
    def _node_to_expr(self, node_address, tree_id):
        node = self.trees[tree_id][node_address]

        if node.nchild == 0:
            return str(node.ft)

        op_name = node.op["name"]
        left_address = lchild_address(node_address)
        left_expr = self._node_to_expr(left_address, tree_id)

        if node.nchild == 1:
            if op_name == "neg":
                return f"(-{left_expr})"
            elif op_name == "inv":
                return f"(1/({left_expr}))"
            elif op_name == "sin":
                return f"sin({left_expr})"
            elif op_name == "cos":
                return f"cos({left_expr})"
            elif op_name == "exp":
                return f"exp({left_expr})"
            elif op_name == "sq":
                return f"({left_expr})^2"
            elif op_name == "cu":
                return f"({left_expr})^3"

        elif node.nchild == 2:
            right_address = rchild_address(node_address)
            right_expr = self._node_to_expr(right_address, tree_id)

            if op_name == "add":
                return f"({left_expr} + {right_expr})"
            elif op_name == "mul":
                return f"({left_expr} * {right_expr})"
            
    # Get symbolic expressions of all trees as strings
    def expression_strings(self):
        exprs = []
        for k in range(len(self.trees)):
            if self.trees[k][0] is None:
                exprs.append("None")
            else:
                exprs.append(self._node_to_expr(0, k))
        return exprs
    
    # Helper 1 for tree printing
    def _node_label(self, node):
        if node.nchild == 0:
            return str(node.ft)
        return str(node.op["name"])

    # Helper 2 for tree printing
    def _build_ascii_tree(self, node_address, tree_id, prefix="", is_last=True):
        node = self.trees[tree_id][node_address]
        if node is None:
            return []

        label = self._node_label(node)
        branch = "└── " if is_last else "├── "
        lines = [prefix + branch + label]

        child_prefix = prefix + ("    " if is_last else "│   ")

        children = []
        if node.nchild >= 1:
            left_address = lchild_address(node_address)
            if left_address < self.max_nodes_per_tree and self.trees[tree_id][left_address] is not None:
                children.append(left_address)
        if node.nchild == 2:
            right_address = rchild_address(node_address)
            if right_address < self.max_nodes_per_tree and self.trees[tree_id][right_address] is not None:
                children.append(right_address)

        for i, child_addr in enumerate(children):
            lines.extend(
                self._build_ascii_tree(
                    child_addr,
                    tree_id,
                    prefix=child_prefix,
                    is_last=(i == len(children) - 1)
                )
            )

        return lines

    # Helper 3 for tree printing: print individual trees
    def _print_tree(self, tree_id):
        if self.trees[tree_id][0] is None:
            print("None")
            return

        root = self.trees[tree_id][0]
        print(self._node_label(root))

        children = []
        if root.nchild >= 1:
            left_address = lchild_address(0)
            if left_address < self.max_nodes_per_tree and self.trees[tree_id][left_address] is not None:
                children.append(left_address)
        if root.nchild == 2:
            right_address = rchild_address(0)
            if right_address < self.max_nodes_per_tree and self.trees[tree_id][right_address] is not None:
                children.append(right_address)

        for i, child_addr in enumerate(children):
            lines = self._build_ascii_tree(child_addr, tree_id, prefix="", is_last=(i == len(children) - 1))
            for line in lines:
                print(line)

    # Prints the full forest in ASCII representation
    def print_forest(self):
        for k in range(len(self.trees)):
            print(f"Tree {k}:")
            self._print_tree(k)
            print()


    # log-marginal-likelihood (beta, sigma^2 integrated out), 
    # posterior mean of beta, 
    # tree design matrix, 
    # posterior hyperparams for (beta, sigma^2)
    def nig_marginal_likelihood(self, X, y, add_intercept = True, prior_params = None):
        # Prior parameters: beta|sigma^2 ~ N(beta0, sigma^2 * V0); sigma^2 ~ Inverse-Gamma(a0 / 2, b0 / 2) 
        if prior_params is not None:
            beta0, V0, a0, b0 = prior_params
        else:
            beta0 = np.zeros(self.ntrees + add_intercept)
            V0 = 10 * np.eye(self.ntrees + add_intercept)
            a0 = 0.05
            b0 = 0.05
        V0_inv = np.linalg.inv(V0)
        
        # Expression Design matrix from standard Design matrix (allcal)
        X_np = np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D matrix of shape n x p.")
        n, p = X.shape
        if p != len(self.features):
            raise ValueError(
                f"X has {p} columns but forest expects {len(self.features)} features."
            )
        TX = np.zeros((n, self.ntrees), dtype=float)
        for k in range(self.ntrees):
            if self.trees[k][0] is None:
                raise ValueError(f"Tree {k} has empty root.")
            TX[:, k] = self._eval_node(0, k, X)
        if add_intercept:
            TX = np.hstack((np.ones((TX.shape[0], 1)), TX))
       
        y = np.asarray(y)
        TXtTX = TX.T @ TX
        TXty = TX.T @ y
        
        # Posterior parameters
        Vn_inv = V0_inv + TXtTX
        Vn = np.linalg.inv(Vn_inv)
        beta_n = Vn @ (V0_inv @ beta0 + TXty)
        a_n = a0 + n
        b_n = b0 + y @ y + beta0 @ V0_inv @ beta0 - beta_n @ Vn_inv @ beta_n
        posterior_params = [beta_n, Vn, a_n, b_n]
        beta_post_mean = beta_n

        # Log-marginal likelihood/ log NIG normalizing constant
        log_det_Vn = np.linalg.slogdet(Vn)[1]
        log_marg_like = - 0.5 * a_n * np.log(b_n / 2) + gammaln(a_n / 2) + 0.5 * log_det_Vn 
        
        return log_marg_like, beta_post_mean, TX, posterior_params
    
    # returns the log prior of a single tree at tree_id
    def log_tree_prior(self, tree_id, alpha_op, alpha_ft):
        alpha_op = np.asarray(alpha_op, dtype=float)
        alpha_ft = np.asarray(alpha_ft, dtype=float)
        
        if len(alpha_op) != len(self.ops):
            raise ValueError("alpha_op must have length len(self.ops).")
        if len(alpha_ft) != len(self.features):
            raise ValueError("alpha_ft must have length len(self.features).")
        if np.any(alpha_op <= 0):
            raise ValueError("alpha_op entries must be strictly positive.")
        if np.any(alpha_ft <= 0):
            raise ValueError("alpha_ft entries must be strictly positive.")
        
        stats = self.tree_statistics(tree_id)
        
        operator_counts = stats["operator_counts"]
        feature_counts = stats["feature_counts"]
        nonterminal_count_by_depth = stats["nonterminal_count_by_depth"]
        terminal_count_by_depth = stats["terminal_count_by_depth"]
        
        # count vectors in the same order as self.ops, self.features
        xi_op = np.array([operator_counts[op["name"]] for op in self.ops], dtype=float)
        kappa_ft = np.array([feature_counts[ft] for ft in self.features], dtype=float)
        
        # log multivariate beta function
        def log_multivariate_beta(vec):
            return np.sum(gammaln(vec)) - gammaln(np.sum(vec))
        
        logp = 0.0
        
        # split/terminal depth contribution
        all_depths = sorted(set(nonterminal_count_by_depth.keys()) | set(terminal_count_by_depth.keys()))
        
        for d in all_depths:
            n_nonterm = nonterminal_count_by_depth.get(d, 0)
            n_term = terminal_count_by_depth.get(d, 0)
            
            p_split = split_prob_func(d, self.maxdepth, alpha=self.alpha, delta=self.delta)
            p_term = 1.0 - p_split
            
            if n_nonterm > 0:
                logp += n_nonterm * np.log(p_split)
            
            if n_term > 0:
                logp += n_term * np.log(p_term)
                
        log_splits = logp
                
        # Dirichlet-collapsed operator/feature terms
        logp += log_multivariate_beta(alpha_op + xi_op)
        logp += log_multivariate_beta(alpha_ft + kappa_ft)
        
        return logp, log_splits
    
    # returns the total log prior of all the trees
    def log_forest_prior(self, alpha_op, alpha_ft):
        logp = 0
        for j in range(self.ntrees):
            lptree, _  = self.log_tree_prior(
                tree_id=j, 
                alpha_op=alpha_op, 
                alpha_ft=alpha_ft
            )
            logp += lptree
        return logp
    
    # returns the log of joint marginal posterior (JMP) of the trees
    def log_JMP(self, alpha_op, alpha_ft, X, y, add_intercept = True, prior_params = None):
        log_forest_prior = self.log_forest_prior(alpha_op=alpha_op, alpha_ft=alpha_ft)
        nig, _, _, _ = self.nig_marginal_likelihood(X=X, y=y, add_intercept=add_intercept, prior_params=prior_params)
        logJMP = log_forest_prior + nig
        
        return logJMP, log_forest_prior, nig
    

# The HierBOSSS MCMC class

class HierBOSSS_MCMC:
    def __init__(self, X, y, K, maxdepth, prior_params=None, add_intercept=True, wts_init=None, wts_prop=None, opset=None, ftset=None):
        # Basic data setup
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.y.ndim != 1:
            raise ValueError("y must be a 1D vector.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Number of rows of X must equal length of y.")
        
        self.n, self.p = self.X.shape
        
        # Model setup
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K: the number of trees must be a positive integer.")
        if not isinstance(maxdepth, int) or maxdepth < 0:
            raise ValueError("maxdepth must be a nonnegative integer.")
        
        self.K = K
        self.maxdepth = maxdepth
        self.add_intercept = bool(add_intercept)
        
        # Operator set / feature set
        if opset is None:
            self.opset = opset = [add, mul, neg, inv, sin, cos, exp, sq, cu]
        else:
            self.opset = opset
        
        if ftset is None:
            self.ftset = [f"x{i}" for i in range(self.p)]
        else:
            self.ftset = list(ftset)
        
        if len(self.ftset) != self.p:
            raise ValueError("Length of ftset (feature set) must equal number of columns in X.")
        
        # Prior parameters
        if prior_params is None:
            alpha_op = np.ones(len(self.opset), dtype=float)
            alpha_ft = np.ones(len(self.ftset), dtype=float)
            alpha = 0.95
            delta = 1.2
            beta0 = np.zeros(self.K + int(self.add_intercept), dtype=float)
            V0 = 10.0 * np.eye(self.K + int(self.add_intercept), dtype=float)
            a0 = 0.05
            b0 = 0.05
        else:
            if len(prior_params) != 8:
                raise ValueError(
                    "prior_params must be a tuple/list of length 8: "
                    "(alpha_op, alpha_ft, alpha, delta, beta0, V0, a0, b0)."
                )

            alpha_op, alpha_ft, alpha, delta, beta0, V0, a0, b0 = prior_params

            alpha_op = np.asarray(alpha_op, dtype=float)
            alpha_ft = np.asarray(alpha_ft, dtype=float)
            beta0 = np.asarray(beta0, dtype=float)
            V0 = np.asarray(V0, dtype=float)
        
        if len(alpha_op) != len(self.opset):
            raise ValueError("alpha_op must have length equal to len(opset).")
        if len(alpha_ft) != len(self.ftset):
            raise ValueError("alpha_ft must have length equal to len(ftset).")
        if np.any(alpha_op <= 0):
            raise ValueError("All entries of alpha_op must be strictly positive.")
        if np.any(alpha_ft <= 0):
            raise ValueError("All entries of alpha_ft must be strictly positive.")
        
        p_beta = self.K + int(self.add_intercept)
        
        if beta0.shape != (p_beta, ):
            raise ValueError(f"beta0 must have shape ({p_beta}, ), got {beta0.shape}.")
        if V0.shape != (p_beta, p_beta):
            raise ValueError(
                f"V0 must have shape ({p_beta}, {p_beta}), got {V0.shape}."
            )
        if a0 <= 0 or b0 <= 0:
            raise ValueError("a0 and b0 must be strictly positive.")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        if delta < 0:
            raise ValueError("delta must be nonnegative.")
        
        self.alpha_op = alpha_op
        self.alpha_ft = alpha_ft
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.beta0 = beta0
        self.V0 = V0
        self.a0 = float(a0)
        self.b0 = float(b0)

        self.prior_params = (
            self.alpha_op,
            self.alpha_ft,
            self.alpha,
            self.delta,
            self.beta0,
            self.V0,
            self.a0,
            self.b0)
        
        # Initialization of weights
        if wts_init is None:
            op_wts_init = np.ones(len(self.opset), dtype=float)
            ft_wts_init = np.ones(len(self.ftset), dtype=float)
        else:
            if len(wts_init) != 2:
                raise ValueError("wts_init must be [op_wts_init, ft_wts_init].")
            op_wts_init = np.asarray(wts_init[0], dtype=float)
            ft_wts_init = np.asarray(wts_init[1], dtype=float)

        if len(op_wts_init) != len(self.opset):
            raise ValueError("Initial operator weights have wrong length.")
        if len(ft_wts_init) != len(self.ftset):
            raise ValueError("Initial feature weights have wrong length.")
        if np.any(op_wts_init < 0) or np.sum(op_wts_init) <= 0:
            raise ValueError("Initial operator weights must be nonnegative with positive sum.")
        if np.any(ft_wts_init < 0) or np.sum(ft_wts_init) <= 0:
            raise ValueError("Initial feature weights must be nonnegative with positive sum.")

        self.wts_init = [
            op_wts_init.astype(float),
            ft_wts_init.astype(float)
        ]
        
        # Proposal weights
        if wts_prop is None:
            op_wts_prop = self.wts_init[0].copy()
            ft_wts_prop = self.wts_init[1].copy()
        else:
            if len(wts_prop) != 2:
                raise ValueError("wts_prop must be [op_wts_prop, ft_wts_prop].")
            op_wts_prop = np.asarray(wts_prop[0], dtype=float)
            ft_wts_prop = np.asarray(wts_prop[1], dtype=float)

        if len(op_wts_prop) != len(self.opset):
            raise ValueError("Proposal operator weights have wrong length.")
        if len(ft_wts_prop) != len(self.ftset):
            raise ValueError("Proposal feature weights have wrong length.")
        if np.any(op_wts_prop < 0) or np.sum(op_wts_prop) <= 0:
            raise ValueError("Proposal operator weights must be nonnegative with positive sum.")
        if np.any(ft_wts_prop < 0) or np.sum(ft_wts_prop) <= 0:
            raise ValueError("Proposal feature weights must be nonnegative with positive sum.")

        self.wts_prop = [
            op_wts_prop.astype(float),
            ft_wts_prop.astype(float)
        ]

        self.fitted = False
    
    # Helper for run_MCMC(), normalizing move weights
    def _normalize_move_weights(self, move_weights):
        move_names = [
            "grow",
            "prune",
            "change_feature",
            "change_operator",
            "subtree_replace",
            "delete_node",
            "insert_node"
        ]

        if move_weights is None:
            move_weights = {m: 1.0 for m in move_names}
        elif isinstance(move_weights, (list, tuple, np.ndarray)):
            if len(move_weights) != len(move_names):
                raise ValueError("Unnamed move_weights must have length 7.")
            move_weights = {m: float(w) for m, w in zip(move_names, move_weights)}
        else:
            move_weights = {k: float(v) for k, v in move_weights.items()}

        for m in move_names:
            move_weights.setdefault(m, 0.0)

        bad = [m for m, w in move_weights.items() if w < 0]
        if bad:
            raise ValueError(f"Move weights must be nonnegative. Bad moves: {bad}")

        total = sum(move_weights.values())
        if total <= 0:
            raise ValueError("At least one move weight must be positive.")

        return move_weights

    # Helper for _sample_move_and_address(), checking valid moves for a tree
    def _available_moves_for_tree(self, forest, tree_id, move_weights):
        stats = forest.tree_statistics(tree_id)
        valid = stats["valid_move_addresses"]

        available = {}
        for move_name, weight in move_weights.items():
            addrs = valid.get(move_name, [])
            if weight > 0 and len(addrs) > 0:
                available[move_name] = {
                    "weight": weight,
                    "addresses": addrs
                }
        return available

    # Helper for update_tree_MH(), randomly selects a move and a node address to perform that chosen move
    def _sample_move_and_address(self, forest, tree_id, move_weights):
        available = self._available_moves_for_tree(forest, tree_id, move_weights)
        if len(available) == 0:
            raise ValueError(f"No valid moves available for tree {tree_id}.")

        move_names = list(available.keys())
        weights = np.array([available[m]["weight"] for m in move_names], dtype=float)
        weights = weights / weights.sum()

        chosen_move = random.choices(move_names, weights=weights, k=1)[0]
        chosen_addr = random.choice(available[chosen_move]["addresses"])

        return chosen_move, chosen_addr, available

    # Helper for update_tree_MH(), applies the supplied move at the supplied node address
    def _apply_move(self, forest, tree_id, move_name, node_address):
        forest_new = copy.deepcopy(forest)

        if move_name == "grow":
            forest_new.grow_node(self.wts_prop, node_address, tree_id)

        elif move_name == "prune":
            forest_new.prune_node(self.wts_prop, node_address, tree_id)

        elif move_name == "change_feature":
            forest_new.exchange_features(self.wts_prop, node_address, tree_id)

        elif move_name == "change_operator":
            forest_new.exchange_operators(self.wts_prop, node_address, tree_id)

        elif move_name == "subtree_replace":
            forest_new.subtree_replace(self.wts_prop, node_address, tree_id)

        elif move_name == "delete_node":
            forest_new.delete_node(node_address, tree_id)

        elif move_name == "insert_node":
            forest_new.insert_node(self.wts_prop, node_address, tree_id)

        else:
            raise ValueError(f"Unknown move {move_name}.")

        return forest_new
    
    # Helper for update_tree_MH(), weight normalization of proposal weights
    def _normalized_prop_weights(self):
        op_wts = np.asarray(self.wts_prop[0], dtype=float)
        ft_wts = np.asarray(self.wts_prop[1], dtype=float)

        op_probs = op_wts / op_wts.sum()
        ft_probs = ft_wts / ft_wts.sum()
        return op_probs, ft_probs
    
    # Helper for update_tree_MH(), mapping the reverse moves
    def _reverse_move(self, move_name):
        rev = {
            "grow": "prune",
            "prune": "grow",
            "change_feature": "change_feature",
            "change_operator": "change_operator",
            "subtree_replace": "subtree_replace",
            "delete_node": "insert_node",
            "insert_node": "delete_node",
        }
        return rev[move_name]
    
    # Helper for update_tree_MH(), gives the log probability of newly generated subtree for grow, subtree_replace, and inser moves
    def _log_subtree_generation_prob(self, subtree, depth):
        op_probs, ft_probs = self._normalized_prop_weights()

        node = subtree["node"]
        if node is None:
            raise ValueError("Subtree root cannot be None.")

        curdepth = depth

        # terminal
        if node.nchild == 0:
            if curdepth >= self.maxdepth:
                ft_idx = self.ftset.index(node.ft)
                return np.log(ft_probs[ft_idx])

            p_split = split_prob_func(curdepth, self.maxdepth, alpha=self.alpha, delta=self.delta)
            p_term = 1.0 - p_split
            ft_idx = self.ftset.index(node.ft)
            return np.log(p_term) + np.log(ft_probs[ft_idx])

        # nonterminal
        if curdepth >= self.maxdepth:
            return -np.inf

        p_split = split_prob_func(curdepth, self.maxdepth, alpha=self.alpha, delta=self.delta)
        op_idx = self.opset.index(node.op)

        logp = np.log(p_split) + np.log(op_probs[op_idx])

        logp += self._log_subtree_generation_prob(subtree["left"], depth + 1)

        if node.nchild == 2:
            logp += self._log_subtree_generation_prob(subtree["right"], depth + 1)

        return logp
    
    # Helper for update_tree_MH(), computes the log probability of a feature draw
    def _log_feature_draw_prob(self, feature_name):
        _, ft_probs = self._normalized_prop_weights()
        ft_idx = self.ftset.index(feature_name)
        return np.log(ft_probs[ft_idx])
    
    # Helper for update_tree_MH(), computes the log probability of an operator draw
    def _log_operator_draw_prob(self, op_dict):
        op_probs, _ = self._normalized_prop_weights()
        op_idx = self.opset.index(op_dict)
        return np.log(op_probs[op_idx])
    
    # Helper for update_tree_MH(), computes the log proposal of a change operator at a given node
    def _log_change_operator_local_prob(self, current_op, new_op):
        op_wts = np.asarray(self.wts_prop[0], dtype=float)
        cur_arity = current_op["type"]

        allowed = np.array([op["type"] == cur_arity for op in self.opset], dtype=bool)
        masked = op_wts.copy()
        for i, ok in enumerate(allowed):
            if not ok:
                masked[i] = 0.0

        cur_idx = self.opset.index(current_op)
        masked[cur_idx] = 0.0

        s = masked.sum()
        if s <= 0:
            return -np.inf

        new_idx = self.opset.index(new_op)
        return np.log(masked[new_idx] / s)
    
    # Helper for update_tree_MH(), computes the exact log proposal of a given move from move dictionary
    def _log_proposal_given_move(self, forest_from, forest_to, tree_id, move_name, move_address):
        stats_from = forest_from.tree_statistics(tree_id)
        valid_from = stats_from["valid_move_addresses"]

        if move_address not in valid_from.get(move_name, []):
            return -np.inf

        n_addr = len(valid_from[move_name])
        logp = -np.log(n_addr)

        # old/new local subtrees
        old_sub = forest_from._get_subtree_copy(move_address, tree_id)
        new_sub = forest_to._get_subtree_copy(move_address, tree_id)

        if move_name == "grow":
            logp += self._log_subtree_generation_prob(new_sub, depth=old_sub["node"].depth)

        elif move_name == "prune":
            new_ft = new_sub["node"].ft
            logp += self._log_feature_draw_prob(new_ft)

        elif move_name == "change_feature":
            old_ft = old_sub["node"].ft
            new_ft = new_sub["node"].ft
            if old_ft == new_ft:
                return -np.inf
            n_choices = len(self.ftset) - 1
            logp += -np.log(n_choices)

        elif move_name == "change_operator":
            old_op = old_sub["node"].op
            new_op = new_sub["node"].op
            logp += self._log_change_operator_local_prob(old_op, new_op)

        elif move_name == "subtree_replace":
            logp += self._log_subtree_generation_prob(new_sub, depth=old_sub["node"].depth)

        elif move_name == "delete_node":
            arity = old_sub["node"].nchild
            if arity == 1:
                logp += 0.0
            elif arity == 2:
                logp += -np.log(2.0)
            else:
                return -np.inf

        elif move_name == "insert_node":
            inserted_op = new_sub["node"].op
            logp += self._log_operator_draw_prob(inserted_op)

            if inserted_op["type"] == 2:
                # old subtree becomes left child; right child is newly grown
                right_sub = new_sub["right"]
                logp += self._log_subtree_generation_prob(right_sub, depth=new_sub["node"].depth + 1)

        else:
            raise ValueError(f"Unknown move {move_name}")

        return logp
        
    
    # Helper for run_MCMC(), for calling and computing log JMP
    def _log_posterior(self, forest):
        log_JMP_val, _, _ = forest.log_JMP(
            alpha_op=self.alpha_op,
            alpha_ft=self.alpha_ft,
            X=self.X,
            y=self.y,
            add_intercept=self.add_intercept,
            prior_params=(self.beta0, self.V0, self.a0, self.b0)
        )
        return log_JMP_val

    # Helper for run_MCMC(), for calling and computing posterior mean of beta
    def _beta_post_mean(self, forest):
        _, beta_mean, _, _ = forest.nig_marginal_likelihood(
            X=self.X,
            y=self.y,
            add_intercept=self.add_intercept,
            prior_params=(self.beta0, self.V0, self.a0, self.b0)
        )
        return beta_mean
    
    # updating a single tree structure using Metropolis-Hastings
    def update_tree_MH(self, forest, tree_id, move_weights):
        move_type, move_address, available = self._sample_move_and_address(
            forest=forest,
            tree_id=tree_id,
            move_weights=move_weights
        )
        
        # normalized move type probability under current state
        move_names_fwd = list(available.keys())
        move_wts_fwd = np.array([available[m]["weight"] for m in move_names_fwd], dtype=float)
        move_probs_fwd = move_wts_fwd / move_wts_fwd.sum()
        log_p_move_fwd = np.log(move_probs_fwd[move_names_fwd.index(move_type)])

        log_post_current = self._log_posterior(forest)

        try:
            forest_prop = self._apply_move(
                forest=forest,
                tree_id=tree_id,
                move_name=move_type,
                node_address=move_address
            )
            log_post_proposed = self._log_posterior(forest_prop)
        except Exception:
            return {
                "forest": copy.deepcopy(forest),
                "accepted": False,
                "move_type": move_type,
                "move_address": move_address,
                "log_mh_ratio": -np.inf,
                "log_post_current": log_post_current,
                "log_post_proposed": -np.inf,
                "log_q_forward": -np.inf,
                "log_q_reverse": -np.inf
            }
        
        # forward proposal
        log_q_forward_local = self._log_proposal_given_move(
            forest_from=forest,
            forest_to=forest_prop,
            tree_id=tree_id,
            move_name=move_type,
            move_address=move_address
        )
        log_q_forward = log_p_move_fwd + log_q_forward_local
        
         # reverse move bookkeeping
        reverse_move = self._reverse_move(move_type)

        available_rev = self._available_moves_for_tree(
            forest=forest_prop,
            tree_id=tree_id,
            move_weights=move_weights
        )

        if reverse_move not in available_rev:
            return {
                "forest": copy.deepcopy(forest),
                "accepted": False,
                "move_type": move_type,
                "reverse_move_type": reverse_move,
                "move_address": move_address,
                "log_mh_ratio": -np.inf,
                "log_post_current": log_post_current,
                "log_post_proposed": log_post_proposed,
                "log_q_forward": log_q_forward,
                "log_q_reverse": -np.inf
            }

        move_names_rev = list(available_rev.keys())
        move_wts_rev = np.array([available_rev[m]["weight"] for m in move_names_rev], dtype=float)
        move_probs_rev = move_wts_rev / move_wts_rev.sum()
        log_p_move_rev = np.log(move_probs_rev[move_names_rev.index(reverse_move)])

        # We use the same address in the reverse move.
        # This is consistent with your local tree rewriting moves.
        log_q_reverse_local = self._log_proposal_given_move(
            forest_from=forest_prop,
            forest_to=forest,
            tree_id=tree_id,
            move_name=reverse_move,
            move_address=move_address
        )
        log_q_reverse = log_p_move_rev + log_q_reverse_local

        log_mh_ratio = (
            log_post_proposed
            - log_post_current
            + log_q_reverse
            - log_q_forward
        )

        accept = False
        if np.isfinite(log_mh_ratio):
            if np.log(np.random.uniform()) <= min(0.0, log_mh_ratio):
                accept = True

        return {
            "forest": forest_prop if accept else copy.deepcopy(forest),
            "accepted": accept,
            "move_type": move_type,
            "reverse_move_type": reverse_move,
            "move_address": move_address,
            "log_mh_ratio": log_mh_ratio,
            "log_post_current": log_post_current,
            "log_post_proposed": log_post_proposed,
            "log_q_forward": log_q_forward,
            "log_q_reverse": log_q_reverse}
        
    # Full MCMC single run
    def run_MCMC(
        self,
        seed=1,
        move_weights=None,
        maxiter=1000,
        burnin=0,
        thin=1,
        show_progress=True,
        progress_queue=None,
        chain_id=None,
        report_every=1
    ):
        if maxiter <= 0:
            raise ValueError("maxiter must be positive.")
        if burnin < 0:
            raise ValueError("burnin must be nonnegative.")
        if thin <= 0:
            raise ValueError("thin must be positive.")

        random.seed(seed)
        np.random.seed(seed)

        move_weights = self._normalize_move_weights(move_weights)
        
        if progress_queue is not None:
            show_progress = False

        # Initialize forest
        current_forest = Forest(
            ntrees=self.K,
            maxdepth=self.maxdepth,
            opset=self.opset,
            ftset=self.ftset,
            alpha=self.alpha,
            delta=self.delta,
            wts_init=self.wts_init
        )

        forests_store = []
        beta_store = []
        logjmp_store = []
        kept_iterations = []

        accepted_hist = []
        move_type_hist = []
        move_address_hist = []
        log_mh_hist = []

        # initial posterior for display
        current_logjmp = self._log_posterior(current_forest)

        iterator = range(1, maxiter + 1)
        if show_progress:
            iterator = tqdm(iterator, total=maxiter, desc="HierBOSSS-MCMC", leave=True)

        total_accept = 0
        total_proposals = 0

        for it in iterator:
            iter_accept = []
            iter_move_type = []
            iter_move_address = []
            iter_log_mh = []

            for tree_idx in range(self.K):
                upd = self.update_tree_MH(
                    forest=current_forest,
                    tree_id=tree_idx,
                    move_weights=move_weights
                )

                current_forest = upd["forest"]
                iter_accept.append(upd["accepted"])
                iter_move_type.append(upd["move_type"])
                iter_move_address.append(upd["move_address"])
                iter_log_mh.append(upd["log_mh_ratio"])

                total_accept += int(upd["accepted"])
                total_proposals += 1

            accepted_hist.append(iter_accept)
            move_type_hist.append(iter_move_type)
            move_address_hist.append(iter_move_address)
            log_mh_hist.append(iter_log_mh)

            current_logjmp = self._log_posterior(current_forest)
            _, log_forest_prior, nig = current_forest.log_JMP(
                alpha_op=self.alpha_op,
                alpha_ft=self.alpha_ft,
                X=self.X,
                y=self.y,
                add_intercept=self.add_intercept,
                prior_params=(self.beta0, self.V0, self.a0, self.b0)
            )
            
            acc_rate = total_accept / max(total_proposals, 1)

            if progress_queue is not None and (it % report_every == 0 or it == maxiter):
                progress_queue.put({
                    "chain_id": chain_id,
                    "seed": seed,
                    "iter": it,
                    "maxiter": maxiter,
                    "JMP": float(current_logjmp),
                    "lognig": float(nig),
                    "lfp": float(log_forest_prior),
                    "acc": float(acc_rate),
                    "done": False
                })

            keep = (it > burnin) and ((it - burnin) % thin == 0)
            if keep:
                forests_store.append(copy.deepcopy(current_forest))
                beta_store.append(self._beta_post_mean(current_forest))
                logjmp_store.append(current_logjmp)
                kept_iterations.append(it)

            if show_progress:
                #acc_rate = total_accept / max(total_proposals, 1)
                #recent_moves = ",".join(str(m) for m in iter_move_type)

                iterator.set_postfix({
                    # "iter": it,
                    "JMP": f"{current_logjmp:.3f}",
                    "lognig": f"{nig:.3f}",
                    "lfp": f"{log_forest_prior:.3f}",
                    "acc": f"{acc_rate:.3f}"
                    # "kept": len(forests_store),
                    # "moves": recent_moves
                })

        if progress_queue is not None:
            progress_queue.put({
                "chain_id": chain_id,
                "seed": seed,
                "iter": maxiter,
                "maxiter": maxiter,
                "done": True
            })
    
        self.result = {
            "forests": forests_store,
            "beta_post_mean": np.array(beta_store),
            "log_jmp": np.array(logjmp_store),
            "accepted": np.array(accepted_hist, dtype=object),
            "move_type": np.array(move_type_hist, dtype=object),
            "move_address": np.array(move_address_hist, dtype=object),
            "log_mh_ratio": np.array(log_mh_hist, dtype=float),
            "kept_iterations": np.array(kept_iterations, dtype=int),
            "seed": seed,
            "move_weights": move_weights
        }
        self.fitted = True
        return self.result
    
    # To run a single chain of MCMC
    def _run_one_chain(
        self,
        seed,
        move_weights=None,
        maxiter=1000,
        burnin=0,
        thin=1,
        show_progress=False
    ):
        model = HierBOSSS_MCMC(
            X = self.X,
            y=self.y,
            K=self.K,
            maxdepth=self.maxdepth,
            prior_params=self.prior_params,
            add_intercept=self.add_intercept,
            wts_init=self.wts_init,
            wts_prop=self.wts_prop,
            opset=self.opset,
            ftset=self.ftset
        )
        
        return model.run_MCMC(
            seed=seed,
            move_weights=move_weights,
            maxiter=maxiter,
            burnin=burnin,
            thin=thin,
            show_progress=show_progress
        )
    
    # To run parallel MCMC chains
    def run_parallel_chains(
        self,
        seeds,
        move_weights=None,
        maxiter=1000,
        burnin=0,
        thin=1,
        n_jobs=None,
        show_progress=True,
        report_every=10
    ):
        import os
        import queue
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import Manager

        if seeds is None or len(seeds) == 0:
            raise ValueError("Provide a nonempty list of seeds.")

        seeds = list(seeds)
        move_weights = self._normalize_move_weights(move_weights)

        if n_jobs is None:
            n_jobs = min(len(seeds), os.cpu_count() or 1)

        manager = Manager()
        progress_queue = manager.Queue()

        worker_args = []
        for chain_id, seed in enumerate(seeds):
            worker_args.append((
                self.X,
                self.y,
                self.K,
                self.maxdepth,
                self.prior_params,
                self.add_intercept,
                self.wts_init,
                self.wts_prop,
                self.opset,
                self.ftset,
                seed,
                move_weights,
                maxiter,
                burnin,
                thin,
                chain_id,
                progress_queue,
                report_every
            ))

        results = []
        errors = []

        bars = None
        if show_progress:
            bars = []
            for chain_id, seed in enumerate(seeds):
                bar = tqdm(
                    total=maxiter,
                    desc=f"Chain {chain_id + 1} | seed={seed}",
                    position=chain_id,
                    leave=True
                )
                bars.append(bar)

        done_count = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_parallel_chain_worker, args) for args in worker_args]

            while done_count < len(seeds):
                try:
                    msg = progress_queue.get(timeout=0.2)
                except queue.Empty:
                    msg = None

                if msg is not None and show_progress:
                    cid = msg["chain_id"]

                    if msg.get("done", False):
                        bars[cid].n = msg["maxiter"]
                        bars[cid].refresh()
                        done_count += 1
                    else:
                        bars[cid].n = msg["iter"]
                        bars[cid].set_postfix({
                            "JMP": f'{msg["JMP"]:.3f}',
                            "lognig": f'{msg["lognig"]:.3f}',
                            "lfp": f'{msg["lfp"]:.3f}',
                            "acc": f'{msg["acc"]:.3f}'
                        })
                        bars[cid].refresh()
                elif msg is not None and not show_progress:
                    if msg.get("done", False):
                        done_count += 1

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    errors.append(str(e))

        if bars is not None:
            for bar in bars:
                bar.close()

        results = sorted(results, key=lambda z: z["seed"])

        self.parallel_result = {
            "chains": results,
            "seeds": [r["seed"] for r in results],
            "n_chains_requested": len(seeds),
            "n_chains_completed": len(results),
            "errors": errors,
            "move_weights": move_weights,
            "maxiter": maxiter,
            "burnin": burnin,
            "thin": thin,
            "n_jobs": n_jobs
        }

        return self.parallel_result
    
    # To summarize the top k forests results from parallel MCMC chains
    def summarize_parallel_topk_forests(self, top_k=20):
        if not hasattr(self, "parallel_result"):
            raise ValueError("Run run_parallel_chains() or run_multiple_chains_serial() first.")

        chains = self.parallel_result["chains"]
        if len(chains) == 0:
            raise ValueError("No chain results found.")

        rows_expr = []
        rows_beta = []

        for chain_idx, chain_res in enumerate(chains):
            forests = chain_res["forests"]
            log_jmp = np.asarray(chain_res["log_jmp"], dtype=float)
            kept_iterations = np.asarray(chain_res["kept_iterations"], dtype=int)
            seed = chain_res.get("seed", None)

            for i, forest in enumerate(forests):
                exprs = forest.expression_strings()

                _, beta_post_mean_from_nig, TX, _ = forest.nig_marginal_likelihood(
                    X=self.X,
                    y=self.y,
                    add_intercept=self.add_intercept,
                    prior_params=(self.beta0, self.V0, self.a0, self.b0)
                )

                yhat = TX @ beta_post_mean_from_nig
                rmse = np.sqrt(np.mean((self.y - yhat) ** 2))

                expr_row = {
                    "chain_id": chain_idx + 1,
                    "chain_seed": seed,
                    "iteration": int(kept_iterations[i]),
                    "log_JMP": float(log_jmp[i]),
                    "RMSE": float(rmse),
                }
                for j in range(self.K):
                    expr_row[f"tree_{j+1}"] = exprs[j]
                rows_expr.append(expr_row)

                beta_row = {
                    "chain_id": chain_idx + 1,
                    "chain_seed": seed,
                    "iteration": int(kept_iterations[i]),
                    "log_JMP": float(log_jmp[i]),
                    "RMSE": float(rmse),
                }

                if self.add_intercept:
                    beta_row["intercept"] = float(beta_post_mean_from_nig[0])
                    for j in range(self.K):
                        beta_row[f"beta_tree_{j+1}"] = float(beta_post_mean_from_nig[j + 1])
                else:
                    for j in range(self.K):
                        beta_row[f"beta_tree_{j+1}"] = float(beta_post_mean_from_nig[j])

                rows_beta.append(beta_row)

        expr_df = pd.DataFrame(rows_expr)
        beta_df = pd.DataFrame(rows_beta)

        order = beta_df["log_JMP"].sort_values(ascending=False).index
        expr_df = expr_df.loc[order].reset_index(drop=True)
        beta_df = beta_df.loc[order].reset_index(drop=True)

        expr_df.insert(0, "rank", np.arange(1, len(expr_df) + 1))
        beta_df.insert(0, "rank", np.arange(1, len(beta_df) + 1))

        if top_k is not None:
            expr_df = expr_df.head(top_k).copy()
            beta_df = beta_df.head(top_k).copy()

        return expr_df, beta_df

    # trace plots for log JMP across MCMC parallel chains
    def plot_parallel_log_jmp(self, figsize=(11, 6), alpha=0.9, linewidth=2.0):
        if not hasattr(self, "parallel_result"):
            raise ValueError("Run run_parallel_chains() or run_multiple_chains_serial() first.")

        chains = self.parallel_result["chains"]
        if len(chains) == 0:
            raise ValueError("No chain results found.")

        plt.figure(figsize=figsize)

        for chain_idx, chain_res in enumerate(chains):
            kept_iterations = np.asarray(chain_res["kept_iterations"], dtype=int)
            log_jmp = np.asarray(chain_res["log_jmp"], dtype=float)

            label = f"Chain {chain_idx + 1}"

            plt.plot(
                kept_iterations,
                log_jmp,
                label=label,
                alpha=alpha,
                linewidth=linewidth
            )

        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("log JMP", fontsize=12)
        plt.title("log JMP Traces Across Parallel Chains", fontsize=14, pad=12)
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.show()

    # To summarize the top k forests results
    def summarize_topk_forests(self, top_k=10):
        if not(self.fitted):
            ValueError("Not yet fitted. Run MCMC before viewing results.")
        forests = self.result["forests"]
        beta_mat = self.result["beta_post_mean"]
        log_jmp = self.result["log_jmp"]
        kept_iterations = self.result["kept_iterations"]

        X = self.X
        y = self.y
        K = self.K

        if len(forests) == 0:
            raise ValueError("No stored forests found in result.")

        rows_expr = []
        rows_beta = []

        for i, forest in enumerate(forests):
            exprs = forest.expression_strings()

            # get TX from nig_marginal_likelihood
            _, beta_post_mean_from_nig, TX, _ = forest.nig_marginal_likelihood(
                X=X,
                y=y,
                add_intercept=self.add_intercept,
                prior_params=[self.beta0, self.V0, self.a0, self.b0]
            )

            yhat = TX @ beta_post_mean_from_nig
            rmse = np.sqrt(np.mean((y - yhat) ** 2))

            expr_row = {}
            for j in range(K):
                expr_row[f"tree_{j+1}"] = exprs[j]
            rows_expr.append(expr_row)

            beta_row = {
                "iteration": int(kept_iterations[i]),
                "log_JMP": float(log_jmp[i]),
                "RMSE": float(rmse),
            }

            if self.add_intercept:
                beta_row["intercept"] = float(beta_post_mean_from_nig[0])
                for j in range(K):
                    beta_row[f"beta_tree_{j+1}"] = float(beta_post_mean_from_nig[j + 1])
            else:
                for j in range(K):
                    beta_row[f"beta_tree_{j+1}"] = float(beta_post_mean_from_nig[j])

            rows_beta.append(beta_row)

        expr_df = pd.DataFrame(rows_expr)
        beta_df = pd.DataFrame(rows_beta)

        order = beta_df["log_JMP"].sort_values(ascending=False).index
        expr_df = expr_df.loc[order].reset_index(drop=True)
        beta_df = beta_df.loc[order].reset_index(drop=True)

        expr_df.insert(0, "rank", range(1, len(kept_iterations) + 1))
        beta_df.insert(0, "rank", range(1, len(kept_iterations) + 1))

        if top_k is not None:
            expr_df = expr_df.head(top_k).copy()
            beta_df = beta_df.head(top_k).copy()

        return expr_df, beta_df

    # trace plot for the log JMP values over MCMC iterations
    def plot_log_jmp_over_iterations(self, figsize=(8, 5)):
        if not(self.fitted):
            ValueError("Not yet fitted. Run MCMC before viewing results.")
        log_jmp = self.result["log_jmp"]
        kept_iterations = self.result["kept_iterations"]

        if len(log_jmp) == 0:
            raise ValueError("No stored log JMP values found in result.")

        plt.figure(figsize=figsize)
        plt.plot(kept_iterations, log_jmp)
        plt.xlabel("Iteration")
        plt.ylabel("log JMP")
        plt.title("log JMP over stored iterations")
        plt.grid(True, alpha=0.3)
        plt.show()

    # summarize the complete results
    def summarize_all_forests(self):
        return self.summarize_topk_forests(top_k=None)
    
    # computing the out-of-sample RMSE using the test dataset corresponding to the max-JMP model obtained during training phase
    def compute_topranked_oos_rmse(mcmc_obj, expr_df, beta_df, X_test, y_test):
        if len(beta_df) == 0:
            raise ValueError("beta_df is empty.")
        if not hasattr(mcmc_obj, "parallel_result"):
            raise ValueError("Run parallel chains first.")

        top_beta_row = beta_df.iloc[0]
        top_expr_row = expr_df.iloc[0]

        target_chain_id = int(top_beta_row["chain_id"])
        target_iteration = int(top_beta_row["iteration"])

        chain_res = mcmc_obj.parallel_result["chains"][target_chain_id - 1]
        kept_iterations = np.asarray(chain_res["kept_iterations"], dtype=int)

        match_idx = np.where(kept_iterations == target_iteration)[0]
        if len(match_idx) == 0:
            raise ValueError("Could not match the top-ranked iteration in stored chain results.")
        match_idx = int(match_idx[0])

        forest_obj = chain_res["forests"][match_idx]

        # posterior mean beta from training data for this forest
        _, beta_post_mean, _, _ = forest_obj.nig_marginal_likelihood(
            X=mcmc_obj.X,
            y=mcmc_obj.y,
            add_intercept=mcmc_obj.add_intercept,
            prior_params=(mcmc_obj.beta0, mcmc_obj.V0, mcmc_obj.a0, mcmc_obj.b0)
        )

        # test design matrix
        n_test_local = X_test.shape[0]
        TX_test = np.zeros((n_test_local, forest_obj.ntrees), dtype=float)
        for k in range(forest_obj.ntrees):
            TX_test[:, k] = forest_obj._eval_node(0, k, X_test)

        if mcmc_obj.add_intercept:
            TX_test = np.hstack((np.ones((n_test_local, 1)), TX_test))

        yhat_test = TX_test @ beta_post_mean
        oos_rmse = np.sqrt(np.mean((y_test - yhat_test) ** 2))

        top_row_info = pd.DataFrame([{
            "rank": int(top_beta_row["rank"]),
            "chain_id": int(top_beta_row["chain_id"]),
            "chain_seed": int(top_beta_row["chain_seed"]),
            "iteration": int(top_beta_row["iteration"]),
            "log_JMP": float(top_beta_row["log_JMP"]),
            "oos_RMSE": float(oos_rmse),
            **({"intercept": float(beta_post_mean[0])} if mcmc_obj.add_intercept else {}),
            **{f"beta_tree_{j+1}": float(beta_post_mean[j + int(mcmc_obj.add_intercept)])
            for j in range(forest_obj.ntrees)},
            **{f"tree_{j+1}": top_expr_row[f"tree_{j+1}"]
            for j in range(forest_obj.ntrees)}
        }])

        return top_row_info, yhat_test, oos_rmse
    
