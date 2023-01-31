import numpy as np
import time
import random
from soph import DEFAULT_FOOTPRINT_RADIUS
import logging


class Node:
    def __init__(self, position, cost_to_root, parent=None):
        self._position = position
        self._cost_to_root = cost_to_root
        self._parent = parent
        self._children = []

    def position(self):
        return self._position

    def cost(self):
        return self._cost_to_root

    def parent(self):
        return self._parent

    def children(self):
        return self._children

    def set_position(self, position):
        self._position = position

    def set_cost(self, cost):
        self._cost_to_root = cost

    def set_parent(self, parent):
        if self._parent is not None:
            self._parent.remove_child(self)
        self._parent = parent
        if self._parent is not None:
            self._parent.add_child(self)

    def add_child(self, child):
        self._children.append(child)

    def remove_child(self, child):
        self._children.remove(child)

    def distance_direct(self, node):
        return np.linalg.norm(self._position - node.position())


class NodeClusters:
    def __init__(self):
        self.clusters = {}

    def nodeToKey(self, node):
        return (int(np.floor(node.position()[0])), int(np.floor(node.position()[1])))

    def getNodes(self, node):
        key = self.nodeToKey(node)
        if key in self.clusters:
            return self.clusters[key]
        return []

    def getNodesAdjacent(self, node, min_radius=1, max_radius=5):
        nodes = []
        key = self.nodeToKey(node)
        radius = 0
        while radius < min_radius or (len(nodes) == 0 and radius < max_radius):
            radius += 1
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (key[0] + i, key[1] + j) in self.clusters:
                        nodes.extend(self.clusters[(key[0] + i, key[1] + j)])
        return nodes

    def addNode(self, node):
        key = self.nodeToKey(node)
        if key in self.clusters:
            self.clusters[key].append(node)
        else:
            self.clusters[key] = [node]

    def totalNodes(self):
        n = 0
        for key in self.clusters:
            n += len(self.clusters[key])
        return n

    def searchSpace(self):
        return len(self.clusters)


class RTRRTstar:
    def __init__(self, init_pos):
        self.root = Node(np.array(init_pos), 0)
        self.Q_r = []
        self.Q_s = []
        self.visited = []
        self.node_clusters = NodeClusters()
        self.node_clusters.addNode(self.root)
        self.k_max = 5
        self.r_s = 0.5
        self.blocked = []
        self.current_path = []
        self.dummy_goal_node = None
        self.path_to_goal = False
        self.map = None

    def setGoalNode(self, goal_position):
        self.dummy_goal_node = Node(goal_position, np.inf)
        self.blocked = []
        self.current_path = []
        self.path_to_goal = False

    def initiate(self, occupancy_map, max_time=1):
        self.map = occupancy_map
        self.setGoalNode(np.array([0, 0]))
        start = time.process_time()
        while time.process_time() - start < max_time:
            self.expandAndRewire()

    # Algorithm 1: RT-RRT*
    def nextIter(
        self, robot_pos, robot_theta, occupancy_map, new_goal=None, max_time=0.1
    ):
        # Update to most up-to-date map
        self.map = occupancy_map
        # Update Goal
        if new_goal is not None:
            self.setGoalNode(new_goal)
        # Expand and Rewire as long as time is left
        start_time = time.process_time()
        while time.process_time() - start_time < max_time:
            self.expandAndRewire(max_time + start_time - time.process_time())
        self.updateNextBestPath()
        while np.linalg.norm(robot_pos - self.current_path[0].position()) < 0.05:
            if len(self.current_path) > 1:
                self.current_path.pop(0)
                self.changeRoot(self.current_path[0])
            elif len(self.current_path) == 1:
                if self.isPathToGoalAvailable():
                    return None, True
                else:
                    return None, False
        diff = self.current_path[0].position() - robot_pos
        diff = diff / np.linalg.norm(diff)
        theta = np.arctan2(diff[1], diff[0])

        # ang_diff = theta - robot_theta
        # if ang_diff > np.pi:
        #     ang_diff -= 2 * np.pi
        # if ang_diff < -np.pi:
        #     ang_diff += 2 * np.pi
        # if abs(ang_diff) > np.pi / 60:
        #     return [
        #         0,
        #         -np.sign(ang_diff) * np.pi / 60,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #     ], False
        # return [
        #     0.5,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        #     0,
        # ], False
        return [
            robot_pos[0] + 0.05 * diff[0],
            robot_pos[1] + 0.05 * diff[1],
            theta,
        ], False

    # Algorithm 2: Tree Expansion-and-Rewiring
    def expandAndRewire(self, max_time=0.01):
        start_time = time.process_time()
        n_new = self.sample()
        adjacent_nodes = self.node_clusters.getNodesAdjacent(n_new)
        if len(adjacent_nodes) == 0:
            return
        n_closest = self.getClosestNeighbor(n_new, adjacent_nodes)
        if self.lineOfSight(n_new, n_closest):
            near = self.findNodesNear(n_new, adjacent_nodes)
            if len(near) < self.k_max or n_closest.distance_direct(n_new) > self.r_s:
                self.addNode(n_new, n_closest, near)
                self.Q_r.insert(0, n_new)
            else:
                self.Q_r.insert(0, n_closest)
            self.rewireRandomNode(start_time, max_time)
        self.rewireFromRoot(start_time, max_time)

        return

    # Algorithm 6: Plan a Path for k Steps
    def updateNextBestPath(self, threshold=0.5, k=100):
        adjacent_nodes = self.node_clusters.getNodesAdjacent(self.dummy_goal_node)
        closest_to_goal = self.getClosestNeighbor(self.dummy_goal_node, adjacent_nodes)
        d = self.dummy_goal_node.distance_direct(closest_to_goal)

        path = []
        if d < threshold and self.lineOfSight(closest_to_goal, self.dummy_goal_node):
            n = closest_to_goal
            while n is not None:
                path.insert(0, n)
                n = n.parent()
            self.current_path = path
            self.path_to_goal = True
            return self.current_path

        path.append(self.root)
        while len(path) < k:
            children = path[-1].children()
            best_child = None
            best_h = np.inf
            for child in children:
                h = child.cost() + self.getHeuristic(child)
                if h < best_h:
                    best_child = child
                    best_h = h
            if best_child is None:
                break
            path.append(best_child)
            if len(best_child.children()) == 0 or all(
                elem in self.blocked for elem in best_child.children()
            ):
                self.blocked.append(best_child)
                break

        if len(self.current_path) == 0 or self.current_path[-1].distance_direct(
            self.dummy_goal_node
        ) > path[-1].distance_direct(self.dummy_goal_node):
            self.current_path = path

        return self.current_path

    def sample(self, alpha=0.1, beta=2):
        p_r = random.uniform(0, 1)
        if p_r > 1 - alpha:
            vec = self.dummy_goal_node.position() - self.root.position()
            p_v = random.uniform(0, 1)
            pos = (
                self.root.position()
                + p_v * vec
                + np.array([random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)])
            )
        elif p_r <= (1 - alpha) / beta or not self.isPathToGoalAvailable():
            pos = self.map.sample_uniform()
        else:
            pos = self.sampleEllipsis()
        return Node(pos, np.inf)

    def sampleEllipsis(self):
        x_goal = self.dummy_goal_node.position()
        x_start = self.root.position()
        c_min = np.linalg.norm(x_start - x_goal)
        c_max = self.current_path[-1].cost() + self.current_path[-1].distance_direct(
            self.dummy_goal_node
        )
        centre = (x_start + x_goal) / 2

        # necessary due to rounding errors
        if c_min > c_max:
            c_min = c_max

        M = (
            1
            / c_max
            * np.array([[x_goal[0] - x_start[0], 0], [x_goal[1] - x_start[1], 0]])
        )
        u, s, vh = np.linalg.svd(M)

        C = u @ np.array([[1, 0], [0, np.linalg.det(u) * np.linalg.det(vh)]]) @ vh

        L = np.array([[c_max / 2, 0], [0, np.sqrt(c_max**2 - c_min**2) / 2]])

        r = np.sqrt(np.random.uniform(0, 1))
        a = np.pi * np.random.uniform(0, 2)

        x_circle = np.array([r * np.cos(a), r * np.sin(a)])

        x_rand = C @ L @ x_circle + centre
        return x_rand

    def getClosestNeighbor(self, node, adjacent_nodes):
        min_dist = np.inf
        closest_neighbor = None
        for n in adjacent_nodes:
            d = node.distance_direct(n)
            if d < min_dist:
                min_dist = d
                closest_neighbor = n
        return closest_neighbor

    def findNodesNear(self, node, adjacent_nodes):
        epsilon = max(
            np.sqrt(
                self.map.free_space()
                * self.k_max
                / (np.pi * self.node_clusters.totalNodes())
            ),
            self.r_s,
        )
        near = []
        for n in adjacent_nodes:
            if node.distance_direct(n) <= epsilon:
                near.append(n)
        return near

    # Algorithm 3: Add Node To Tree
    def addNode(self, n_new, n_closest, near):
        n_min = n_closest
        c_min = n_closest.cost() + n_closest.distance_direct(n_new)
        for n in near:
            c_new = n.cost() + n.distance_direct(n_new)
            if c_new < c_min and self.lineOfSight(n, n_new):
                n_min = n
                c_min = c_new
        n_new.set_parent(n_min)
        n_new.set_cost(c_min)
        self.node_clusters.addNode(n_new)

    # Algorithm 4: Rewire Random Nodes
    def rewireRandomNode(self, start_time, max_time):
        while len(self.Q_r) > 0 and time.process_time() - start_time < max_time:
            n_r = self.Q_r.pop(0)
            adjacent_nodes = self.node_clusters.getNodesAdjacent(n_r)
            near = self.findNodesNear(n_r, adjacent_nodes)
            c_r = self.costRecursive(n_r)
            for n in near:
                c_old = self.costRecursive(n)
                c_new = c_r + n_r.distance_direct(n)
                if c_new < c_old and self.lineOfSight(n_r, n):
                    n.set_parent(n_r)
                    n.set_cost(c_new)
                    self.Q_r.append(n)
        return

    # Algorithm 5: Rewire From the Tree Root
    def rewireFromRoot(self, start_time, max_time, force_reset=False):
        if len(self.Q_s) == 0 or force_reset:
            self.Q_s = [self.root]
            self.visited = [self.root]

        while len(self.Q_s) > 0 and time.process_time() - start_time < max_time:
            n_s = self.Q_s.pop(0)
            adjacent_nodes = self.node_clusters.getNodesAdjacent(n_s)
            near = self.findNodesNear(n_s, adjacent_nodes)
            c_s = self.costRecursive(n_s)
            for n in near:
                c_old = self.costRecursive(n)
                c_new = c_s + n_s.distance_direct(n)
                if c_new < c_old and self.lineOfSight(n_s, n):
                    n.set_parent(n_s)
                    n.set_cost(c_new)
                if n not in self.visited:
                    self.Q_s.append(n)
                    self.visited.append(n)

        return

    def costRecursive(self, node):
        cost = 0
        n_curr = node
        while n_curr.parent() is not None:
            if n_curr.parent().cost() == np.inf:
                n_curr.set_cost(np.inf)
                return np.inf
            cost += n_curr.parent().distance_direct(n_curr)
            n_curr = n_curr.parent()
        node.set_cost(cost)
        return cost

    def getHeuristic(self, node):
        if node in self.blocked:
            return np.inf
        return node.distance_direct(self.dummy_goal_node)

    def changeRoot(self, new_root):
        new_root.set_cost(0)
        new_root.set_parent(None)

        self.root.set_parent(new_root)
        self.root.set_cost(self.root.distance_direct(new_root))

        self.root = new_root
        self.Q_s = []
        self.blocked = []
        return

    def isPathToGoalAvailable(self):
        return self.path_to_goal

    def lineOfSight(self, n_1, n_2, base_radius=DEFAULT_FOOTPRINT_RADIUS):

        if self.map.check_if_free(n_1.position(), base_radius) is False:
            return False
        if self.map.check_if_free(n_2.position(), base_radius) is False:
            return False

        diff = n_2.position() - n_1.position()
        pos = n_1.position()
        n = int(np.ceil(np.linalg.norm(diff) / base_radius))
        for i in range(0, n):
            pos = pos + diff / n
            if self.map.check_if_free(pos, base_radius) is False:
                return False

        return True
