import numpy as np
import logging


class Node:
    def __init__(self, position, parent):
        self.position = position
        self.parent = parent

    def set_parent_recursive(self, new_parent):
        """
        Set parent of this node to a new parent, as well as set the current parents parent to this node.
        """
        if self.parent is not None:
            self.parent.set_parent_recursive(self)
        self.parent = new_parent

    def get_cost(self):
        """
        Get Cost to reach root of this node.
        Cost is calculated by taking euclidean distances between each parent node until root is reached.
        """
        current_node = self
        nextnode = self.parent
        total_cost = 0
        while nextnode is not None:
            cost = np.linalg.norm(current_node.position - nextnode.position)
            total_cost += cost
            current_node = nextnode
            nextnode = nextnode.parent

        return total_cost

    def get_path(self):
        """
        Get path to reach this node from root.
        If self is root, an empty list is returned.
        """
        node = self
        nodes = []
        while node.parent is not None:
            nodes.insert(0, node)
            node = node.parent
        return nodes


class NavGraph:
    def __init__(self, start_position):
        self.root = Node(start_position, None)
        self.nodes = [self.root]

    def add_node_at(self, position, parent=None):
        """
        Add new node at specified position.
        If no parent is given, current root is set as parent of new node.
        """
        new_node = Node(position, self.root if parent is None else parent)
        self.nodes.append(new_node)
        return new_node

    def move_root(self, new_root):
        """
        Move the root of the graph to specified node.
        """
        self.root = new_root
        self.root.set_parent_recursive(None)

    def get_closest_node(self, position, map=None):
        """
        Get the closest node to a given position.
        If a map is given, line of sight is taken into account and nodes with no line of sight are disregarded.
        """
        closest_node = None
        best_dist = np.inf
        for node in self.nodes:
            dist = np.linalg.norm(node.position - position)
            if dist < best_dist:
                if map is not None:
                    pos_in_map = map.m_to_px(position)
                    node_in_map = map.m_to_px(node.position)
                    if not map.line_of_sight(node_in_map, pos_in_map):
                        continue
                closest_node = node
                best_dist = dist
        return closest_node

    def get_near_nodes(self, node, map, max_radius=2.0):
        """
        Get list of nodes with Line of Sight to given node within a maximum radius.
        """
        near = []
        for n in self.nodes:
            if n == node or n == node.parent:
                continue
            # TODO Add check if line of sight
            if np.linalg.norm(n.position - node.position) <= max_radius:
                n_in_map = map.m_to_px(n.position)
                node_in_map = map.m_to_px(node.position)
                if not map.line_of_sight(n_in_map, node_in_map):
                    continue
                near.append(n)
        return near

    def rewire_with_new_node(self, new_node, map, max_radius=2.0):
        """
        Rewire the graph with a new node that was just added.
        """
        near_nodes = self.get_near_nodes(new_node, map, max_radius)
        for n in near_nodes:
            self.rewire_two_nodes(new_node, n)

    def rewire_two_nodes(self, base_node, node_to_rewire):
        """
        Rewire two nodes. The base node becomes the new parent.
        The node_to_rewire recursively gets rewired as long as the cost to reach root is better.
        """
        current_cost = base_node.get_cost() + np.linalg.norm(
            base_node.position - node_to_rewire.position
        )
        total_cost = node_to_rewire.get_cost()
        last_node = base_node
        current_node = node_to_rewire
        while current_cost < total_cost:
            print(current_cost)
            print(total_cost)
            cost = np.linalg.norm(current_node.position - current_node.parent.position)
            next_node = current_node.parent
            current_node.parent = last_node
            last_node = current_node
            current_node = next_node
            current_cost += cost
            total_cost -= cost

    def update_with_robot_pos(self, robot_pos, map, min_dist=1.0):
        """
        Update the graph with a new robot position.
        If the current root is the closest to the new position, a new node is added and root is moved.
        If a different node is closest, a new node is added and the graph is rewired.
        """
        dist = np.linalg.norm(self.root.position - robot_pos)
        original_dist = dist
        closest_node = self.get_closest_node(robot_pos, map)
        previous_root = None
        if closest_node is not self.root and closest_node is not None:
            closest_dist = np.linalg.norm(robot_pos - closest_node.position)
            if closest_dist < dist:
                previous_root = self.root

                logging.info(
                    "Moving Root to "
                    + f"{closest_node.position[0]:.2f}, {closest_node.position[1]:.2f}"
                )
                self.move_root(closest_node)
                dist = closest_dist

        if dist > min_dist:
            logging.info(
                "Adding nav_node at " + f"{robot_pos[0]:.2f}, {robot_pos[1]:.2f}"
            )
            new_node = self.add_node_at(robot_pos)
            self.move_root(new_node)
            self.rewire_with_new_node(new_node, map, 1.1 * original_dist)
        if previous_root is not None:
            self.rewire_two_nodes(closest_node, previous_root)
