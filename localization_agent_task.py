import math
from queue import PriorityQueue
from typing import Dict, Tuple, Generator, Callable, Any

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
    return np.all((pos >= 0) & (pos < occ_map.shape))


def is_valid_free_pos_on_occ_map(occ_map: np.ndarray, pos: np.ndarray) -> bool:
    if not (is_valid_pos_on_occ_map(occ_map, pos)):
        return False
    return occ_map[tuple(pos)] == 0.


def empty_neighbours(occ_map: np.ndarray, cell: Tuple[int, int]) -> Generator[Tuple[int, int], None, None]:
    for neighbor_pos in [np.sum(list(zip(cell, pos_alteration)), axis=1) for pos_alteration in
                         [(0, 1), (1, 0), (-1, 0), (0, -1)]]:
        if is_valid_pos_on_occ_map(occ_map, neighbor_pos) and not (occ_map[tuple(neighbor_pos)]):
            yield tuple(x for x in neighbor_pos)


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


def normal_pdf(mean_, std_, x: np.ndarray) -> np.ndarray:
    return np.exp(-np.square((x - mean_) / std_) / 2) / (std_ * np.sqrt(2 * math.pi))


def normpdf(mean, sd, x):
    var = sd ** 2
    denom = (2 * math.pi * var) ** .5
    num = np.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom


class LocalizationMap:
    def __init__(self, environment):
        # We could compute lidar for all cells and thus tell where most probably we are?
        # (by measuring diff between our lidar and the lidar of all cells)
        # since it's 3d vector, using kd-trees or sth like that would be efficient?
        # cons:
        # - not applicable in continuous case
        # - not usable if someone changes angles of lidar during run
        # - not too efficient?
        occ_map = environment.gridmap

        self.empty_cells_indices = np.array(list(
            x for x in zip(*np.where(environment.gridmap == 0.))))  # 5k x 2
        self.lidars = np.array(list(
            environment.ideal_lidar(environment.rowcol_to_xy(tuple(idx)))[1] for idx in self.empty_cells_indices
        ))
        self.probabilities = np.full((self.empty_cells_indices.shape[0],), 1 / self.empty_cells_indices.shape[0],
                                     dtype='float')

        self.eta = 1.  # TODO: how to compute this?

        self.measurement_pdf: Callable[[np.ndarray], np.ndarray] = lambda x: np.multiply.reduce(normpdf(1.,
                                                                                                        environment.lidar_stochasticity,
                                                                                                        x), axis=1)

        # self.movement_pdf: Callable[[np.ndarray], np.ndarray] = lambda x: 1 - environment.position_stochasticity

        self.environment = environment

    def get_most_probable_position(self) -> Tuple[int, int]:
        return tuple(self.empty_cells_indices[np.argmax(self.probabilities)])

    def position_update_by_motion_model(self, delta: np.ndarray) -> None:
        """
        :param delta: Movement taken by agent in the previous turn.
        It should be one of [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        """ TODO: your code goes here """
        next_probabilities = self.probabilities * self.environment.position_stochasticity

        for i, cell_index in enumerate(self.empty_cells_indices):
            prev_cell_index = cell_index - delta

            if is_valid_free_pos_on_occ_map(self.environment.gridmap, prev_cell_index):
                next_probabilities[i] += (1 - self.environment.position_stochasticity) * self.probabilities[
                    list(zip(*np.where(np.all(self.empty_cells_indices == cell_index - delta, axis=1))))[
                        0]]

        self.probabilities = next_probabilities

    def position_update_by_measurement_model(self, distances: np.ndarray) -> None:
        """
        Updates the probabilities of agent position using the lidar measurement information.
        :param distances: Noisy distances from current agent position to the nearest obstacle.
        """
        pdf_result = self.measurement_pdf(
            distances / self.lidars) * self.probabilities
        pdf_result /= np.sum(
            pdf_result)  # since pdf returns very small probabilities (we're in a continuous distribution),
        # we have to normalize probabilities to sum to 1
        # TODO: take into account that we couldn't make the move because we're not at a given location and there's a wall?
        self.probabilities = self.eta * pdf_result

    def position_update(self, distances: np.ndarray, delta: np.ndarray = None):
        self.position_update_by_motion_model(delta)
        self.position_update_by_measurement_model(distances[1])


class LocalizationAgent:
    def __init__(self, environment):
        self.environment = environment
        self.localization_map = LocalizationMap(environment)
        self.position_probability_matrix = np.zeros_like(environment.gridmap)
        self._update_position_probability_matrix()
        self.cur_pos2_best_move = self._get_best_move_dict()

    def _squeeze(self, x: Tuple[int, int]) -> Tuple[int, int]:
        return tuple(int(y / 10) for y in x)

    def _unsqueeze(self, x: Tuple[int, int]) -> Tuple[int, int]:
        return tuple(int(y * 10) for y in x)

    def _get_best_move_dict(self) -> Dict[Tuple[int, int], np.ndarray]:
        occ_map = np.ones(tuple(int(x / 10) for x in self.environment.gridmap.shape))
        for ind in np.ndindex(*occ_map.shape):
            occ_map[ind] = self.environment.gridmap[self._unsqueeze(ind)]

        result = dict()

        tgt = self._squeeze(self.environment.xy_to_rowcol(self.environment.goal_position))
        for ind in np.ndindex(*occ_map.shape):
            if occ_map[ind] == 0:
                path = a_star_search(occ_map, ind, tgt)
                next_node = path[ind] if ind != tgt else tgt
                delta = np.array(next_node) - np.array(ind)
                for i in range(10):
                    for j in range(10):
                        result[tuple(np.array(self._unsqueeze(ind)) + np.array([i, j]))] = delta
        return result

    def _update_position_probability_matrix(self):
        self.position_probability_matrix[self.localization_map.empty_cells_indices[:, 0],
                                         self.localization_map.empty_cells_indices[:, 1]] \
            = self.localization_map.probabilities

    def _add_maze_to_position_probability_matrix(self):
        self.position_probability_matrix[self.localization_map.empty_cells_indices[:, 0],
                                         self.localization_map.empty_cells_indices[:, 1]] += 0.2 * np.max(
            self.position_probability_matrix)

    def _get_next_step(self) -> np.ndarray:
        most_probable_cur_pos = self.localization_map.get_most_probable_position()
        return self.cur_pos2_best_move[most_probable_cur_pos]

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        # TODO: why won't we do just measurement and then position model updates
        # and instead docstring suggests to do measurement -> position model -> measurement

        # TODO: use lidar multiple times and take a mean?
        distances = self.environment.lidar()[1]

        self.localization_map.position_update_by_measurement_model(distances[1])
        next_step = self._get_next_step()

        print("Actual position: {}".format(list(self.environment.xy_to_rowcol(self.environment.position()))))
        print("Predicted position: {}".format(list(self.localization_map.get_most_probable_position())))
        print("Next step: {}".format(next_step))

        self.environment.step(tuple(next_step))

        distances = self.environment.lidar()[1]
        self.localization_map.position_update(distances, next_step)

        # self.localization_map.position_update_by_motion_model(next_step)

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of current agent position
        """
        self._update_position_probability_matrix()
        self._add_maze_to_position_probability_matrix()
        return self.position_probability_matrix


if __name__ == "__main__":
    maze = generate_maze((11, 11))
    env = Environment(
        maze,
        lidar_angles=3,
        resolution=1 / 11 / 10,
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
            plt.savefig('./tmp/map.png')
            plt.close(plt.gcf())

            cv2.imshow('map', cv2.imread('./tmp/map.png'))
            cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
