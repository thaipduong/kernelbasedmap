import numpy as np
import matplotlib.pyplot as plt
from rtree import index
import kernels



class Fastron(object):
    def __init__(self, kernel = kernels.rbf_kernel, update_argminF=True, remove_redundant=True):
        self.kernel = kernel
        self.update_argminF = update_argminF
        self.remove_redundant = remove_redundant
        self.rtree_index = index.Index()
        self.gamma = 5

    # Calculate kernel
    def G(self, x, y):
        y = np.array([y[0], y[1]])
        x = np.array([x[0], x[1]])
        return self.kernel(x,y, gamma=self.gamma)

    # Train and update the support vectors based on new local observations.
    def train(self, iter_max=100, alpha=None, local_observations = None):
        # Weights alpha
        if alpha is None:
            alpha = {}

        # Calculate score F for the observations. Approximate F using 10 nearest support vectors.
        F = {}
        for nb in local_observations:
            F[nb] = 0
            nearest_alpha = self.rtree_index.nearest((nb[0], nb[1], nb[0], nb[1]), 10, objects=True)
            na_len = 0
            for na in nearest_alpha:
                na_len = na_len +1
                a = (na.bbox[0], na.bbox[1])
                F[nb] = F[nb] + self.G(nb,a)*alpha[a]

        # Start training
        for iter in range(iter_max):
            r_plus = 1.5
            r_minus = 1

            # Find argmin y_i*F_i
            if local_observations is not None:
                if self.update_argminF:
                    min = None
                    local_min_idx = None
                    for nb in local_observations:
                        nb_fy = F[nb]*local_observations[nb]
                        f_sign = np.sign(F[nb])
                        if f_sign == 0.0:
                            f_sign = 1.0
                        if min is None:
                            min = nb_fy
                            local_min_idx = nb
                        elif min > nb_fy or (min == nb_fy and f_sign*local_observations[nb] < 0):
                            min = nb_fy
                            local_min_idx = nb
                    ind_list = [local_min_idx]
                else:
                    ind_list = local_observations
            else:
                return

            # Update weights
            correct_prediction = True
            for i in ind_list:
                y_predict = np.sign(F[i])
                if y_predict == 0.0:
                    y_predict = 1.0
                if y_predict != local_observations[i]:

                    correct_prediction = False
                    r = r_plus if local_observations[i] == 1 else r_minus
                    delta_alpha = r * local_observations[i] - y_predict

                    if i in alpha:
                        alpha[i] = alpha[i] + delta_alpha
                    else:
                        # Add new support vectors to Rtree.
                        alpha[i] = delta_alpha
                        self.rtree_index.insert(i[0]*1000 + i[1], (i[0], i[1], i[0], i[1]))
                    # Update score F based on new weights alpha.
                    for nb in local_observations:
                        F[nb] = F[nb] + delta_alpha * self.G(nb,i)

            # Remove redundant support points
            if self.remove_redundant:
                for m in alpha.copy():
                    if m not in local_observations:
                        continue
                    margin_m = local_observations[m]*(F[m] - alpha[m])
                    if margin_m > 0:
                        for nb in local_observations:
                            F[nb] = F[nb] - alpha[m] * self.G(nb, m)
                        alpha.pop(m)
                        self.rtree_index.delete(m[0]*1000 + m[1], (float(m[0]), float(m[1]), float(m[0]), float(m[1])))
            if correct_prediction:
                print("Fastron finished at iteration " + str(iter))
                break
        return alpha, F

    # Calculate the score F of a point x_test
    def calculate_score(self, alpha, x_test):
        score = 0
        for a in alpha:
            score = score + alpha[a]*self.G(x_test, a)
        return score

    # Predict occupancy labels for a set of points X_test
    def predict(self, X, Y, alpha, X_test):
        f_predict = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        for i in range(X_test.shape[0]):
            x_test = X_test[i,:]
            f_predict[i][0] = self.calculate_score(alpha, x_test)

        y_predict = np.sign(f_predict)
        return y_predict, f_predict

    # Calculate the proposed upper bound on the score.
    def predict_upperbound(self, alpha_dict, X_test):
        X = np.array(list(alpha_dict.keys()), dtype=np.float32)

        alpha = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            alpha[i] = alpha_dict[(X[i, 0], X[i, 1])]
        f_predict_plus = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        f_predict_minus = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        f_predict_upperbound = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        plus_apha = alpha[alpha > 0]
        plus_apha_total = np.sum(plus_apha)
        for i in range(X_test.shape[0]):
            x_test = X_test[i,:]
            G_row = np.zeros([X.shape[0], 1])
            for j in range(X.shape[0]):
                if alpha[j] >= 0:
                    continue
                G_row[j] = self.kernel(X[j, :], x_test, gamma=self.gamma)
            f_predict_minus[i][0] = np.max(G_row)
            minus_idx_min = np.argmax(G_row)
            G_row = np.zeros([X.shape[0], 1])
            for j in range(X.shape[0]):
                if alpha[j] <= 0:
                    continue
                G_row[j] = self.kernel(X[j, :], x_test, gamma=self.gamma)
            f_predict_plus[i][0] = np.max(G_row)
            f_predict_upperbound[i][0] = plus_apha_total*f_predict_plus[i][0] + alpha[minus_idx_min]*f_predict_minus[i][0]

        y_predict = np.sign(f_predict_upperbound)
        return y_predict, f_predict_upperbound

    # Collision checking for a line x(t) = x_A + v*t
    # tighter_bound = True for the better but slower bound (O(M^2) complexity)
    # tighter_bound = False for the looser but faster bound with (O(M) complexity)
    def check_line(self, alpha_dict, A, v, tighter_bound = True):
        # Support vectors and weights
        X = np.array(list(alpha_dict.keys()))
        alpha = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            alpha[i] = alpha_dict[(X[i,0], X[i,1])]

        x_test = A
        G_row = np.zeros([X.shape[0], 1])
        dist = np.zeros([X.shape[0], 1])
        # Find the closest positive and negative support vectors to the point A.
        for j in range(X.shape[0]):
            G_row[j] = self.kernel(X[j, :], x_test, gamma=self.gamma)
            dist[j] = np.linalg.norm(X[j, :] - x_test) ** 2
        min_dist_minus = None
        min_idx_minus = None
        for j in range(X.shape[0]):
            if alpha[j] >= 0:
                continue
            if min_dist_minus == None:
                min_dist_minus = dist[j][0]
                min_idx_minus = j
            elif min_dist_minus > dist[j]:
                min_dist_minus = dist[j][0]
                min_idx_minus = j

        alpha_minus = np.abs(alpha[min_idx_minus])
        total_plus = np.sum(alpha[alpha > 0])
        t_u = None
        for j in range(X.shape[0]):
            if alpha[j] <= 0:
                continue

            if tighter_bound:
                temp_max = None
                for k in range(X.shape[0]):
                    if alpha[k] >= 0:
                        continue
                    beta_plus = (np.log(np.abs(alpha[k])) - np.log(total_plus))/self.gamma
                    temp1 =  beta_plus + dist[j] - dist[k]
                    temp3 = -X[k, :] + X[j, :]
                    temp2 = 2* np.matmul(v, np.transpose(temp3))
                    # If term2 <=0, there is no limit on t.
                    if temp2 <= 0:
                        temp2 = 0.0000000001
                    temp = temp1 / temp2
                    if temp_max == None:
                        temp_max = temp
                    elif temp_max <= temp:
                        temp_max = temp
                if t_u == None:
                    t_u = temp_max
                elif t_u >= temp_max:
                    t_u = temp_max
            else:
                beta_plus = (np.log(np.abs(alpha_minus)) - np.log(total_plus)) / self.gamma
                temp1 = beta_plus + dist[j] - dist[min_idx_minus]
                temp3 = -X[min_idx_minus, :] + X[j, :]
                temp2 = 2* np.matmul(v, np.transpose(temp3))
                # If term2 <=0, there is no limit on t
                if temp2 <=0:
                    temp2 = 0.00000000001
                temp = temp1 / temp2
                if t_u == None:
                    t_u = temp
                elif t_u >= temp:
                    t_u = temp
        if t_u == None:
            t_u = 1
        return t_u


    # Find the radius of the free ball a point x_test
    # tighter_bound = True for the better but slower bound (O(M^2) complexity)
    # tighter_bound = False for the looser but faster bound with (O(M) complexity)
    def check_free_radius(self, X, alpha, x_test, tighter_bound = False):
        G_row = np.zeros([X.shape[0], 1])
        dist = np.zeros([X.shape[0], 1])
        for j in range(X.shape[0]):
            G_row[j] = self.kernel(X[j, :], x_test, gamma=self.gamma)
            dist[j] = np.linalg.norm(X[j, :] - x_test) ** 2
        min_dist_minus = None
        min_idx_minus = None
        for j in range(X.shape[0]):
            if alpha[j] >= 0:
                continue
            if min_dist_minus == None:
                min_dist_minus = dist[j][0]
                min_idx_minus = j
            elif min_dist_minus > dist[j]:
                min_dist_minus = dist[j][0]
                min_idx_minus = j

        alpha_minus = np.abs(alpha[min_idx_minus])
        total_plus = np.sum(alpha[alpha > 0])
        radius = None
        for j in range(X.shape[0]):
            if alpha[j] <= 0:
                continue

            if tighter_bound:
                temp_max = None
                for k in range(X.shape[0]):
                    if alpha[k] >= 0:
                        continue
                    beta_plus = (np.log(np.abs(alpha[k])) - np.log(total_plus)) / self.gamma
                    temp1 = beta_plus + dist[j] - dist[k]
                    temp2 = 2 * np.linalg.norm(X[k, :] - X[j, :])
                    temp = temp1 / temp2
                    if temp_max == None:
                        temp_max = temp
                    elif temp_max < temp:
                        temp_max = temp
                if radius == None:
                    radius = temp_max
                elif radius > temp_max:
                    radius = temp_max
            else:
                beta_plus = (np.log(np.abs(alpha_minus)) - np.log(total_plus)) / self.gamma
                temp1 = beta_plus + dist[j] - dist[min_idx_minus]
                temp2 = 2*np.linalg.norm(X[min_idx_minus, :] - X[j, :])
                temp = temp1/temp2
                if radius == None:
                    radius = temp
                elif radius > temp:
                    radius = temp
        return radius

    # Plotting decision boundary
    def plot_decision_boundary(self, X, Y, alpha, x_min= [-6, -6], x_max=[6, 6], fig=None, ax=None, show_data = False):
        x0 = np.linspace(x_min[0], x_max[0])
        x1 = np.linspace(x_min[1], x_max[1])
        x0mesh, x1mesh = np.meshgrid(x0, x1)
        x0mesh_flattened = x0mesh.flatten()
        x1mesh_flattened = x1mesh.flatten()
        X_grid = np.vstack((x0mesh_flattened, x1mesh_flattened))
        X_grid = np.transpose(X_grid)
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        colors = ['green' if l == -1. else 'red' for l in Y]
        if show_data:
            s = [60 if a in alpha else 0 for a in range(len(X[:, 0]))]
            ax.scatter(X[:, 0], X[:, 1], color='b', s=s)
            ax.scatter(X[:, 0], X[:, 1], color=colors, s=20)
        y_predict, f_predict = self.predict(X, Y, alpha, X_grid)
        ax.contour(x0mesh, x1mesh,
                   f_predict.reshape(x0mesh.shape), levels=[0], cmap="cool")

    # Plot the inflated boundary generated from the upper bound.
    def plot_upperbound_boundary(self, X, Y, alpha, x_min= [-6, -6], x_max=[6, 6], fig=None, ax=None, show_data=False):
        x0 = np.linspace(x_min[0], x_max[0])
        x1 = np.linspace(x_min[1], x_max[1])
        x0mesh, x1mesh = np.meshgrid(x0, x1)
        x0mesh_flattened = x0mesh.flatten()
        x1mesh_flattened = x1mesh.flatten()
        X_grid = np.vstack((x0mesh_flattened, x1mesh_flattened))
        X_grid = np.transpose(X_grid)
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if show_data:
            s = [60 if (X[i,0], X[i,1]) in alpha else 0 for i in range(X.shape[0])]
            c = ['b' if Y[i,0] < 0 else 'r' for i in range(X.shape[0])]
            ax.scatter(X[:, 0], X[:, 1], color=c, s=s)

        y_predict, f_predict_upperbound = self.predict_upperbound(alpha, X_grid)
        CS = ax.contour(x0mesh, x1mesh,
                   f_predict_upperbound.reshape(x0mesh.shape),levels=[0],  cmap="cool_r", linestyles= 'dashed')
