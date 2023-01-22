from queue import PriorityQueue
from typing import Dict, Tuple, Generator

import cv2
import numpy as np
from matplotlib import pyplot as plt
from environment import Environment
from utils import generate_maze


def manhattan_dist(vec1: Tuple[int, int], vec2: Tuple[int, int]) -> int:
    """
    Compute distance between two points represented as 2d tuples.
    """
    return sum(abs(x - y) for x, y in zip(vec1, vec2))


def is_valid_pos_on_occ_map(occ_map: np.ndarray, pos: np.ndarray) -> bool:
    for dim in range(occ_map.ndim):
        if pos[dim] < 0 or pos[dim] >= occ_map.shape[dim]:
            return False

    return True


def empty_neighbours(occ_map: np.ndarray, cell: Tuple[int, int]) -> Generator[Tuple[int, int], None, None]:
    for neighbor_pos in [np.sum(list(zip(cell, pos_alteration)), axis=1) for pos_alteration in
                         [(0, 1), (1, 0), (-1, 0), (0, -1)]]:
        if is_valid_pos_on_occ_map(occ_map, neighbor_pos) and not (occ_map[tuple(neighbor_pos)]):
            yield tuple(x for x in neighbor_pos)


# TODO: write tests for that
def a_star_search(occ_map: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Dict[
    Tuple[int, int], Tuple[int, int]]:
    """
    Implements the A* search with heuristic function being distance from the goal position.
    :param occ_map: Occupancy map, 1 – field is occupied, 0 – is not occupied.
    :param start: Start position from which to perform search
    :param end: Goal position to which we want to find the shortest path
    :return: The dictionary containing at least the optimal path from start to end in the form:
        {start: intermediate, intermediate: ..., almost: goal}
    """
    Node = Tuple[int, int]

    fringe = PriorityQueue()
    prev_node_on_best_path: Dict[Node, Node] = dict()
    lowest_cost_for_node: Dict[Node, int] = {start: 0}
    fringe.put((manhattan_dist(start, end), (start, 0)))  # f, (node, cost to get to it)

    while (cur_node := fringe.get())[1][0] != end:
        cur_node_h, (cur_node_pos, dist_to_cur_node_from_start) = cur_node

        for neighbour_pos in empty_neighbours(occ_map, cur_node_pos):
            cost_for_neighbour = dist_to_cur_node_from_start + 1
            if lowest_cost_for_node.get(neighbour_pos, occ_map.size ** 2) > cost_for_neighbour:
                fringe.put((manhattan_dist(neighbour_pos, end) + cost_for_neighbour,
                            (neighbour_pos,
                             cost_for_neighbour)))
                lowest_cost_for_node[neighbour_pos] = cost_for_neighbour
                prev_node_on_best_path[neighbour_pos] = cur_node_pos

    ans: Dict[Node, Node] = dict()
    next_node = end

    while next_node != start:
        prev_node = prev_node_on_best_path.get(next_node)
        ans[prev_node] = next_node
        next_node = prev_node

    return ans


class LocalizationMap:
    def __init__(self, environment):
        """ TODO: your code goes here """

    def position_update_by_motion_model(self, delta: np.ndarray) -> None:
        """
        :param delta: Movement taken by agent in the previous turn.
        It should be one of [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        """ TODO: your code goes here """

    def position_update_by_measurement_model(self, distances: np.ndarray) -> None:
        """
        Updates the probabilities of agent position using the lidar measurement information.
        :param distances: Noisy distances from current agent position to the nearest obstacle.
        """
        """ TODO: your code goes here """

    def position_update(self, distances: np.ndarray, delta: np.ndarray = None):
        self.position_update_by_motion_model(delta)
        self.position_update_by_measurement_model(distances)


class LocalizationAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of current agent position
        """
        """ TODO: your code goes here """


if __name__ == "__main__":
    maze = generate_maze((11, 11))
    env = Environment(
        maze,
        lidar_angles=3,
        resolution=1/11/10,
        agent_init_pos=None,
        goal_position=(0.87, 0.87),
        position_stochasticity=0.5
    )
    agent = LocalizationAgent(env)

    while not env.success():
        agent.step()

        if env.total_steps % 10 == 0:
            plt.imshow(agent.visualize())
            plt.colorbar()
            plt.savefig('/tmp/map.png')
            plt.close(plt.gcf())

            cv2.imshow('map', cv2.imread('/tmp/map.png'))
            cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
