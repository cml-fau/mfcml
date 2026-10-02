# Mathematical Foundations of Control and Machine Learning

**Friedrich-Alexander-Universität Erlangen-Nürnberg — Department of Mathematics**
Master's level course, Winter semester 2026.

| | |
|---|---|
| **Lecturer** | Prof. Dr. Enrique Zuazua — Chair in Applied Analysis (Alexander von Humboldt Professorship) |
| **Co-lecturer** | Daniel López — <dani.lopez@fau.de> |
| **Evaluation** | Oral presentation (100%); attendance is taken into account |

This repository holds the lecture notebooks and the slides for the student presentations.

## Contents

- [About the course](#about-the-course)
- [Learning objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Course material](#course-material)
- [Installation](#installation)
- [Running the notebooks](#running-the-notebooks)
- [Exporting to HTML](#exporting-to-html)
- [Bibliography](#bibliography)

## About the course

The course provides a unified mathematical perspective on control theory and machine learning. It
introduces key concepts from the analysis of dynamical systems governed by ordinary and partial
differential equations (ODEs and PDEs) — controllability, observability and stability — and builds
on them to show how control-theoretic and optimization-based methods explain the dynamics of
state-of-the-art machine learning systems: deep neural networks, federated learning frameworks,
reinforcement learning and large language models (LLMs).

## Learning objectives

By the end of the course students will have developed a solid understanding of the mathematical and
dynamical foundations underlying control and learning systems. They will be able to

- analyse the optimization and stability properties of deep learning algorithms,
- explain how control-theoretic principles extend to large-scale models such as LLMs,
- deliver clear and rigorous scientific presentations of technical concepts and research results.

## Prerequisites

Strongly recommended: basic knowledge of calculus, linear algebra, ODEs and PDEs. The code is
written in Python with PyTorch, but no prior exposure to either is assumed — the notebooks keep the
code to the shortest fragment that illustrates the mathematics.

## Course material

Each notebook in [`notebooks/`](notebooks/) is self-contained: the mathematics is developed in the
text and the code is the smallest runnable confirmation of it.

| # | Notebook | Topics |
|---|---|---|
| 01 | [Introduction to Machine Learning & Deep Learning](notebooks/01_intro_to_ml.ipynb) | Risk minimization, representations, linear regression, gradient descent, autodiff and backpropagation, neural networks |
| 02 | [Neural ODEs](notebooks/02_neuralodes.ipynb) | Residual networks as Euler steps, numerical ODE solvers, learning dynamics and representations |
| 03 | [Reinforcement Learning](notebooks/03_reinforcement_learning.ipynb) | Bellman equation and value iteration, Hamilton–Jacobi–Bellman, Q-learning, exploration vs. exploitation, Pontryagin's maximum principle |
| 04 | [Optimization and gradient descent](notebooks/04_optimization.ipynb) | Descent lemma, convergence in the convex and non-convex case, momentum, stochastic gradients and the noise floor |
| 05 | [Control Theory](notebooks/05_control_theory.ipynb) | Kalman rank condition, controllability Gramian, pole placement, LQR and the Riccati equation, model predictive control |
| 06 | [Diffusion models](notebooks/06_diffusion_models.ipynb) | The heat equation as noising, Feynman–Kac, Ornstein–Uhlenbeck forward process, reverse-time SDE and probability-flow ODE, score matching |
| 07 | [Transformers](notebooks/07_transformers.ipynb) | Self-attention, multi-head attention, positional encoding, the transformer block, masked language modelling |

Slides for the student presentations go in [`presentations/`](presentations/).

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
`uv.lock`, and PyTorch is installed from the CPU-only index — everything in the course runs on a
laptop without a GPU.

```bash
git clone https://github.com/dani2442/Course-Mathematical-Foundations-of-Control-and-Machine-Learning.git
cd Course-Mathematical-Foundations-of-Control-and-Machine-Learning
uv sync
```

This creates a `.venv/` with Python ≥ 3.11, PyTorch, NumPy, Matplotlib and Jupyter.

<details>
<summary>Without uv</summary>

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install numpy matplotlib jupyter ipykernel
```

</details>

## Running the notebooks

```bash
uv run jupyter lab notebooks/
```

In VS Code, open a notebook and select `.venv/bin/python` as the kernel instead.

To execute a notebook end to end from the command line:

```bash
uv run jupyter nbconvert --execute --inplace notebooks/01_intro_to_ml.ipynb
```

Every notebook opens with a setup cell that seeds the random number generators and fixes the
plotting style, so figures and numbers reproduce exactly.

## Exporting to HTML

`./export-html.sh` renders the notebooks to standalone HTML pages. With no arguments it exports all
of them:

```bash
./export-html.sh                                # all notebooks
./export-html.sh notebooks/05_control_theory.ipynb   # just one
```

Cell tags control what the export shows — `remove-cell` drops a cell, `remove-input` keeps only its
output, `remove-output` keeps only its source. See `jupyter_nbconvert_config.py`.

## Bibliography

1. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.
2. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
3. C. F. Higham and D. J. Higham, "Deep learning: An introduction for applied mathematicians," *SIAM Review*, 61(4) (2019), 860–891.
4. L. Bottou, F. E. Curtis, and J. Nocedal, "Optimization methods for large-scale machine learning," *SIAM Review*, 60(2) (2018), 223–311.
5. J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. Springer, 2006.
6. J.-M. Coron, *Control and Nonlinearity*, Mathematical Surveys and Monographs, Vol. 136. American Mathematical Society, 2007.
7. D. Ruiz-Balet and E. Zuazua, "Neural ODE control for classification, approximation and transport," *arXiv preprint* arXiv:2104.05278, 2021.
8. E. Zuazua, "Propagation, observation, and control of waves approximated by finite difference methods," *SIAM Review*, 47(2) (2005), 197–243.
9. E. Zuazua, "Controllability and observability of partial differential equations: Some results and open problems," in *Handbook of Differential Equations: Evolutionary Equations*, Vol. 3, North-Holland, 2006, pp. 527–621.
