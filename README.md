**This repository contains Python code for the paper ["Autonomous Navigation in Unknown Environments with Sparse Kernel-based Occupancy Mapping"](https://thaipduong.github.io/kernelbasedmap/)**

# SCOPE

For simplicity, the code is for a 10x10 simulated environment instead of the ROS environment used in the paper. However, the main algorithms are the same.

# DEPENDENCIES

The code depends on the following software and packages:
python3.x, numpy, matplotlib, queue, rtree

# COMMAND TO RUN THE CODE
```
python main.py
```

# FILES
## Important code
1) main.py: code for autonomous mapping and navigation algorithm
2) perceptron.py: code related to main contributions: Fastron and collision checking algorithms.
3) robot.py: code for collecting observations and retraining kernel perceptron
## Others
4) env.py: code to generate a simulated environment
5) kernels.py: code for multiple kernel functions.
6) astar.py: code for A*
7) viz.py: code for visualization

# VISUALIZATION EXPLAINED
1) Start cell: top left corner. 
2) Goal cell: bottom right corner.
3) Green triangle: robot.
4) Blue boundary: decision boundary by Fastron score.
5) Magenta dashed boundary: inflated boundary by the upper bound. 
6) Blue arrows: A* path.
7) Yellow arrows: traveled path.

# TEST
The code has been tested on Ubuntu 16.04 and Python 3.6.

