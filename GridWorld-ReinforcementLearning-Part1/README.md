# GridWorld Value Iteration

## Overview
This project implements **Value Iteration** for a **4x4 GridWorld** using the **Bellman equation**. The goal is to find the optimal state-value function for an agent trying to reach the terminal state in the bottom-right corner while minimizing penalties.

## Problem Setup
- The environment is a **4x4 GridWorld** (16 states).
- The agent starts at the **top-left corner** (state `0`).
- The terminal state is the **bottom-right corner** (state `15`).
- The agent can move **up, down, left, or right** with equal probability.
- Each move incurs a reward of **-1**, except for reaching the terminal state, which has a reward of **0**.
- The objective is to compute the **optimal value function** using **Value Iteration**.

## Implementation Details
- The Bellman equation is applied iteratively:

  \[
  V(s) = \max_a \sum_{s'} P(s' | s, a) \left[R(s, a, s') + \gamma V(s') \right]
  \]

- The transition probability is **equal for all actions** (1/4 probability per move).
- The value function is updated iteratively **until convergence** (when the max value change < `1e-4`).
- **No discounting** is used (`gamma = 1.0`).
- **Boundary conditions** are handled to prevent moving out of the grid.

## Installation & Usage
### Prerequisites
- Python 3.x
- NumPy

### Run the Code
Clone the repository and execute the script:
```bash
python gridworld_value_iteration.ipynb
```

## Expected Output
- The final optimal value function (example output):
[[-59.42, -57.42, -54.28, -51.71]
 [-57.42, -54.56, -49.71, -45.13]
 [-54.28, -49.71, -40.85, -29.99]
 [-51.71, -45.13, -29.99,  0.00]]

- Higher values indicate better states for reaching the goal.
- The values decrease as the distance from the goal increases.
