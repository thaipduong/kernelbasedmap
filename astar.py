from queue import PriorityQueue

#
# A* implementation.
#

def astar(start, goal, alpha = None, classifier = None):
    start.g = 0
    start.f = start.g + start.h(goal)
    frontier = PriorityQueue()
    frontier.put(start)

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            while current.parent is not None:
                parent = current.parent
                parent.child = current
                current = parent
            return True

        # Since PriorityQueue in python doesn't support removing nodes, there might be duplicates in the queue.
        # We skip those visited nodes here.
        if current.visited:
            continue

        current.visited = True

        for neighbor in current.neighbors:
            if neighbor.visited:
                continue
            new_g = current.g + current.cost(neighbor, alpha, classifier)
            if new_g < neighbor.g:
                neighbor.g = new_g
                neighbor.f = new_g + neighbor.h(goal)
                frontier.put(neighbor)
                neighbor.inqueue = True
                neighbor.parent = current

    return False
