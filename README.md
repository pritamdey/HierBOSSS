# HierBOSSS: Hierarchical Bayesian Operator-induced Symbolic Regression Trees for Structural Learning of Scientific Expressions

[![Python](https://img.shields.io/badge/Python-3.13.5-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Forks](https://img.shields.io/github/forks/Roy-SR-007/HierBOSSS)](https://github.com/Roy-SR-007/HierBOSSS/network)
[![Repo Size](https://img.shields.io/github/repo-size/Roy-SR-007/HierBOSSS)](https://github.com/Roy-SR-007/HierBOSSS)
[![Last Commit](https://img.shields.io/github/last-commit/Roy-SR-007/HierBOSSS)](https://github.com/Roy-SR-007/HierBOSSS/commits/main)
[![Issues](https://img.shields.io/github/issues/Roy-SR-007/HierBOSSS)](https://github.com/Roy-SR-007/HierBOSSS/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/Roy-SR-007/HierBOSSS)](https://github.com/Roy-SR-007/HierBOSSS/pulls)

<p align="center">
  <img src="hierbosss_tree.gif" alt="HierBOSSS_logo" width="800"/>
</p>

This repository holds the source code and implementation of **HierBOSSS** for Bayesian structural learning of scientific symbolic expressions proposed in Roy, S., Dey, P., Pati, D., & Mallick, B. K. (2025), *Hierarchical Bayesian Operator-induced Symbolic Regression Trees for Structural Learning of Scientific Expressions*.

---

## Developers and Maintainers

**Somjit Roy**  
Department of Statistics  
Texas A&M University, College Station, TX, USA  

📧 Email: [sroy_123@tamu.edu](mailto:sroy_123@tamu.edu)  
🌐 Website: [https://roy-sr-007.github.io](https://roy-sr-007.github.io)

**Pritam Dey**  
Department of Statistics  
Texas A&M University, College Station, TX, USA  

📧 Email: [pritam.dey@tamu.edu](mailto:pritam.dey@tamu.edu)  
🌐 Website: [https://pritamdey.github.io](https://pritamdey.github.io)

---

## Overview

We develop a hierarchical Bayesian framework for symbolic regression that represents scientific laws as ensembles of tree-structured symbolic expressions endowed with a regularized tree prior. This coherent probabilistic formulation enables full posterior inference via an efficient Markov chain Monte Carlo algorithm, yielding a balance between predictive accuracy and structural parsimony. To guide symbolic model selection, we develop a marginal posterior–based criterion adhering to the Occam’s window principle.
<br><br>

<figure align="center">
  <img src="assets/intro_pic.png" alt="symbolic_tree_representation" width="400"/>
  <figcaption><em>HierBOSSS bridges the gap between SciML and Statistical AI in context of symbolic regression.</em></figcaption>
</figure>

<br><br>

<figure align="center">
  <img src="assets/symbolic_tree_representation.png" alt="symbolic_tree_representation" width="800"/>
  <figcaption><em>Figure 1: Symbolic tree representation of scientific expressions.</em></figcaption>
</figure>

<br><br>


**HierBOSSS** models symbolic expressions through an ensemble of symbolic tree-structured scientific expressions, regarded as the symbolic forest component. Conjugate priors are assigned to model regression parameters, while a regularizing prior is designed for the individual symbolic tree structures. To perform inference from the HierBOSSS-induced posterior distribution, we develop an efficient Metropolis-within-partially-collapsed Gibbs Markov chain Monte Carlo (MCMC) sampling algorithm.

<br><br>

---
