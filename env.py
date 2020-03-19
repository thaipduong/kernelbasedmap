import numpy as np
import perceptron
import kernels

# Generate grid environment randomly:
# 0 = free, 1 = blocked
# The probability of a cell being blocked is input p
def grid_env_gen(p, x_size, y_size, fixed_start_goal = True, fixed = True):
    if fixed:
        # Generate a fixed environment instead of a random one.
        data = np.array([[0., 1., 1., 0., 0., 0., 0., 0., 0., 1.,],
                         [0., 0., 1., 0., 0., 0., 1., 1., 0., 0.,],
                         [0., 0., 0., 1., 0., 1., 1., 0., 1., 0.,],
                         [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,],
                         [1., 0., 0., 0., 0., 1., 0., 0., 1., 0.,],
                         [1., 1., 1., 0., 0., 0., 0., 1., 0., 0.,],
                         [1., 0., 1., 0., 0., 1., 1., 1., 0., 1.,],
                         [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,],
                         [0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,]])
    else:
        data = np.random.choice([0, 1.], size=(x_size, y_size), p=[1-p, p])

    # By default the robot starts at the top left conner and tries to reach the bottom right conner.
    if not fixed_start_goal:
        while True:
            x = np.random.randint(0, x_size)
            y = np.random.randint(0, y_size)
            if data[x, y] == 0:
                start_cell = [x, y]
                data[x, y] = 0
                break
        while True:
            x = np.random.randint(0, x_size)
            y = np.random.randint(0, y_size)
            if data[x, y] == 0:
                goal_cell = [x, y]
                data[x, y] = 0
                break
    else:
        data[0,0] = 0
        data[x_size - 1, y_size - 1] = 0
        start_cell = [0,0]
        goal_cell = [x_size - 1, y_size - 1]
    return data, start_cell, goal_cell


# Class Node for cell. Each node contains information for A* to work.
class Node:
    def __init__(self, x, y):
        # Node info
        self.x = x
        self.y = y
        self.data_idx = None

        # Motion model
        self.neighbors = []
        # Observation model
        self.observation_nb = []
        # Cost parameters
        self.collided = []
        self.penalty = 100
        self.step_cost = 1

        # A* parameters
        self.parent = None
        self.child = None
        self.visited = False
        self.inqueue = False
        self.g = float('inf')
        self.f = float('inf')
        self.observed = False

        # For visualization
        self.blocked_prob = 0.0

    # Define cost function for A* based on the support vectors
    # If colliding, cost = inf else step_cost
    def cost(self, next_node, alpha = None, classifier = None):
        if alpha is not None and classifier is not None:
            t_uA = classifier.check_line(alpha, [self.x, self.y],
                                        [next_node.x - self.x, next_node.y - self.y],
                                        tighter_bound=True)
            t_uB = classifier.check_line(alpha, [next_node.x, next_node.y],
                                        [self.x -next_node.x, self.y - next_node.y],
                                        tighter_bound=True)
            if t_uA + t_uB >= 1:
                return self.step_cost
            else:
                return float('inf')

    # l2 norm heuristic for A*
    def h(self, goal):
        return np.sqrt((self.x - goal.x) ** 2 + (self.y - goal.y) ** 2)

    # For priority queue
    def __lt__(self, other):
        return self.f < other.f

    def reset_astar(self):
        self.parent = None
        self.child = None
        self.visited = False
        self.inqueue = False
        self.g = float('inf')
        self.f = float('inf')
