import numpy as np
import perceptron
import kernels
import env

# Class for robot.
class Robot:
    def __init__(self, grid_env, start_node = None, allow_diag = True):
        # Need ground truth map for observation model
        self.grid_env = grid_env
        self.x_size = grid_env.shape[0]
        self.y_size = grid_env.shape[1]

        # How far the robot can observe its neighbor cells in Manhattan distance.
        self.observation_range = 3

        # Build a grid graph for A*
        self.V = self.build_graph(allow_diag)
        if start_node is None:
            self.current_node = self.V[0][0]
        else:
            self.current_node = start_node

        # Variables for Fastron
        x0 = np.linspace(0, self.x_size - 1, self.x_size)
        x1 = np.linspace(0, self.y_size - 1, self.y_size)
        x0mesh, x1mesh = np.meshgrid(x0, x1)
        x0mesh = x0mesh.flatten()
        x1mesh = x1mesh.flatten()
        self.X = np.vstack((x0mesh, x1mesh))
        self.X = np.transpose(self.X)

        for i in range(self.X.shape[0]):
            x = self.X[i, :].astype(int)
            self.V[x[0]][x[1]].data_idx = i

        # All cells are assigned labeled -1 (free) as we do not have any observations yet.
        self.Y = -np.ones([self.X.shape[0], 1])
        self.alpha = None
        self.F = None

        # Fastron classifier with gaussian kernel
        self.classifier = perceptron.Fastron(kernel=kernels.rbf_kernel, update_argminF=True, remove_redundant=True)

    # Calculate Perceptron score F(x)
    def calculate_score(self, X, alpha, kernel = kernels.rbf_kernel):
        score = np.zeros([X.shape[0], 1], dtype=np.float64)
        for cell in range(len(score)):
            for a in alpha:
                a_vec = [a[0], a[1]]
                score[cell] = score[cell] + alpha[a]*kernel(X[cell,:],a_vec, self.classifier.gamma)
        return score

    # Observe the environment and retrain the kernel perceptron model as a map
    def observe_and_update_map(self):
        local_observations = {}
        local_observations[(self.current_node.x, self.current_node.y)] = self.Y[self.current_node.data_idx]

        for observed_neighbor in self.current_node.observation_nb:
            observed_neighbor.observed = True

            # Observe the neighbors
            if self.grid_env[observed_neighbor.x, observed_neighbor.y] > 0.5:
                self.Y[observed_neighbor.data_idx] = 1
            local_observations[(observed_neighbor.x, observed_neighbor.y)] = self.Y[observed_neighbor.data_idx][0]

            # Augmented data for this observation model
            for augmented in observed_neighbor.observation_nb:
                local_observations[(augmented.x, augmented.y)] = self.Y[augmented.data_idx][0]

        # Update the support vectors with the new observations.
        self.alpha, self.F = self.classifier.train(iter_max=200, alpha=self.alpha, local_observations=local_observations)

        # Calculate the score of every points on the grid, and set their occupancy label based on the sign of the score.
        # This is to visualize the observed regions of the environment.
        score = self.calculate_score(self.X, self.alpha)
        for i in range(len(score)):
            x = self.X[i, :].astype(int)
            self.V[x[0]][x[1]].blocked_prob = 0 if score[i] < 0 else 1


    # Generate graph based on the env matrix. Edges are added between neighboring nodes according to the environment.
    def build_graph(self, allow_diag=True):
        V = [[env.Node(i, j) for j in range(self.grid_env.shape[1])] for i in range(self.grid_env.shape[0])]
        for i in range(self.grid_env.shape[0]):
            for j in range(self.grid_env.shape[1]):
                for k in range(-self.observation_range,self.observation_range + 1):
                    for l in range(-self.observation_range,self.observation_range + 1):
                        if 0 <= i + k < self.grid_env.shape[0] and 0 <= j + l < self.grid_env.shape[1] and (k != 0 or l != 0) and ((k**2 + l**2) < 2 or allow_diag):
                            # This defines the neighbors of a cell for motion model.
                            # By default, a robot can potentially go to any of its 8 neighbors.
                            V[i][j].neighbors.append(V[i+k][j+l])
                        if 0 <= i + k < self.grid_env.shape[0] and 0 <= j + l < self.grid_env.shape[1] and (k != 0 or l != 0):
                            # This defines the set of neighbors a robot can observe from its location.
                            V[i][j].observation_nb.append(V[i+k][j+l])
        return V

    # This function reset A* state of all the cells for replanning.
    def reset_astar_states(self):
        for i in range(self.x_size):
            for j in range(self.y_size):
                self.V[i][j].reset_astar()