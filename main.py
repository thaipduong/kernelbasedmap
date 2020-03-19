import astar
import env
import robot
import numpy as np
import matplotlib.pyplot as plt
import viz

# Generate random grid environment and visualize it.
grid_env, start_cell, goal_cell = env.grid_env_gen(0.1, 10, 10, fixed=True)
# Visualize the ground truth map
viz.plot_grid(np.transpose(grid_env), brightness = 0.5)
plt.savefig("./figs/gt.png")
#plt.show()

# Build graph of nodes based on the environment.
bot = robot.Robot(grid_env, allow_diag=False)
start_node = bot.V[start_cell[0]][start_cell[1]]
goal_node = bot.V[goal_cell[0]][goal_cell[1]]

# Parameters.
N = 100
current_node = start_node
previous_node = None
# Stores robot locations in the past for visualization.
history = []
fig, ax = plt.subplots()

for i in range(N):
    print("Time step:", i)
    bot.current_node = current_node

    # Observe and update the support vectors.
    bot.observe_and_update_map()

    # Run A*
    bot.reset_astar_states()
    is_path_found = astar.astar(current_node, goal_node, bot.alpha, bot.classifier)
    if is_path_found:
        print("Found path!")
    else:
        print("Path not found!")

    history.append(current_node)
    # Visualize the environment after A*
    ax.clear()
    # Show path
    grid_env_with_path = viz.graph2grid(grid_env, bot.V, start_node, goal_node)
    # Show perceptron decision boundary and its inflated boundary
    bot.classifier.plot_upperbound_boundary(bot.X, bot.Y, bot.alpha, x_min=[-0.5, -0.5],
                                            x_max=[bot.x_size - 0.5, bot.y_size - 0.5], fig=fig, ax=ax,
                                            show_data=False)
    bot.classifier.plot_decision_boundary(bot.X, bot.Y, bot.alpha, x_min=[-0.5, -0.5],
                                            x_max=[bot.x_size - 0.5, bot.y_size - 0.5], fig=fig, ax=ax,
                                            show_data=False)
    viz.plot_grid(grid_env_with_path, start_node, goal_node, history, fig=fig, ax=ax)
    plt.savefig("./figs/"+str(i)+".png")
    plt.pause(0.5)


    if current_node == goal_node:
        break

    # If a path is found, move to the next cell on the path
    # If not, step back.
    if is_path_found:
        previous_node = current_node
        current_node = current_node.child
    else:
        current_node = previous_node

plt.show()