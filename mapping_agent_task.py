import math
from enum import Enum
from typing import Tuple, Optional, Generator, Collection, Iterable

import cv2
import numpy as np
from matplotlib import pyplot as plt

from environment import Environment
from utils import generate_maze
from localization_agent_task import a_star_search


class OccupancyMap:
    def __init__(self, environment):
        self.occ_map = 0.5 * np.ones_like(
            environment.gridmap)  # TODO: gridmap's 10x actual gridmap, we could make our occ_map smaller...
        self.environment = environment
        self.prob_cell_taken_if_observed_taken = 0.53

    def point_update(self, pos: Tuple[int, int], distance: Optional[float], total_distance: Optional[float],
                     occupied: bool) -> None:
        """
        Update regarding noisy occupancy information inferred from lidar measurement.
        :param pos: rowcol grid coordinates of position being updated
        :param distance: optional distance from current agent position to the :param pos: (your solution don't have to use it)
        :param total_distance: optional distance from current agent position to the final cell from current laser beam (your solution don't have to use it)
        :param occupied: whether our lidar reading tell us that a cell on :param pos: is occupied or not
        """
        # TODO: make use of distance and use knowledge on noise distribution to compute prob_cell_taken_if_observed_taken more accurately?
        prob = self.prob_cell_taken_if_observed_taken if occupied else 1 - self.prob_cell_taken_if_observed_taken
        correction = np.log(
            prob / (1 - prob))
        if not occupied:
            if distance > 10:
                correction *= 1 / (distance / 5)  # dimnish changes for far cells
        self.occ_map[pos] += np.log(
            prob / (1 - prob))

    @staticmethod
    def _ray_trace_passed_cells_revised(angle: float,
                                        distance_rowcol: float,
                                        pos_rowcol: Tuple[int, int]) -> Iterable[Tuple[float, Tuple[int, int]]]:
        distance_x = math.floor(np.abs(np.cos(angle)) * distance_rowcol)
        pos = np.array(pos_rowcol, dtype=np.float64)
        func = np.floor if np.cos(angle) > 0 else np.ceil
        passed_cells = {(0., tuple(pos.astype(int)))}
        for i in range(1, distance_x):
            pos[0] = pos_rowcol[0] + i * np.sign(np.cos(angle))
            pos[1] = pos_rowcol[1] + i * np.sin(angle)
            passed_cells.add((np.linalg.norm(pos), tuple(func(pos).astype(int))))
            if np.all(func(pos) == pos):
                passed_cells.add((np.linalg.norm(pos), tuple((pos + np.array([0, 1])).astype(int))))
                passed_cells.add((np.linalg.norm(pos), tuple((pos + np.array([1, 0])).astype(int))))

        distance_y = math.floor(np.abs(np.sin(angle)) * distance_rowcol)
        pos = np.array(pos_rowcol, dtype=np.float64)
        func = np.floor if np.sin(angle) > 0 else np.ceil
        for i in range(1, distance_y):
            pos[0] = pos_rowcol[0] + i * np.cos(angle)
            pos[1] = pos_rowcol[1] + i * np.sign(np.sin(angle))
            passed_cells.add((np.linalg.norm(pos), tuple(func(pos).astype(int))))
            if np.all(func(pos) == pos):
                passed_cells.add((np.linalg.norm(pos), tuple((pos + np.array([0, 1])).astype(int))))
                passed_cells.add((np.linalg.norm(pos), tuple((pos + np.array([1, 0])).astype(int))))
        return [x for x in passed_cells if np.all((np.array(x[1]) < 110) & (np.array(x[1]) >= 0))]

    @staticmethod
    def _ray_stop_cells(angle: float, pos_rowcol: Tuple[int, int], distance_rowcol: np.ndarray) -> Generator[
        Tuple[int, int], None, None]:
        pos_stopped = tuple((np.array(pos_rowcol) + np.ceil(
            np.array([np.cos(angle) * distance_rowcol, np.sin(angle) * distance_rowcol]))).astype(int))
        if np.all(np.array(pos_stopped) < 110):
            yield pos_stopped

    def map_update(self, pos: Tuple[float, float], angles: np.ndarray, distances: np.ndarray) -> None:
        """
        :param pos: current agent position in xy in [0; 1] x [0; 1]
        :param angles: angles of the beams that lidar has returned
        :param distances: distances from current agent position to the nearest obstacle in directions :param angles:
        """
        pos_rowcol = self.environment.xy_to_rowcol(pos)
        distances_rowcol = distances * self.environment.gridmap.shape[0]

        # for each angle, compute which cells it passes through
        # and at which it stops
        for angle, distance_rowcol in zip(angles, distances_rowcol):
            for dist, passed_cell in self._ray_trace_passed_cells_revised(angle, distance_rowcol, pos_rowcol):
                self.point_update(passed_cell, dist, None, occupied=False)

            for stop_cell in self._ray_stop_cells(angle, pos_rowcol, distance_rowcol):
                self.point_update(stop_cell, distance_rowcol, None, occupied=True)


class Direction(Enum):
    UP = (1, 0)
    DOWN = (-1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class MappingAgent:

    def __init__(self, environment):
        self.occ_map = OccupancyMap(environment)
        self.environment = environment

        self.current_dirs: Tuple[Direction, Direction] = (Direction.UP, Direction.RIGHT)
        self.turn_remaining_moves = 0
        self.turn = False
        self.init_find_wall_right = True
        self.init_moves = 100

    def clockwise_turn(self, direction: Direction):
        if direction == Direction.UP:
            return Direction.RIGHT
        elif direction == Direction.RIGHT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.LEFT
        else:
            return Direction.UP

    def anti_clockwise_turn(self, direction: Direction):
        if direction == Direction.UP:
            return Direction.LEFT
        elif direction == Direction.RIGHT:
            return Direction.UP
        elif direction == Direction.DOWN:
            return Direction.RIGHT
        else:
            return Direction.DOWN

    def can_go_in_direction(self,
                            direction: Tuple[int, int]):
        pos = np.array(self.environment.xy_to_rowcol(self.environment.position()))
        dir = np.array(direction)
        next_ = pos + dir
        if np.any(next_ >= 110) or np.any(next_ < 0):
            return False
        return self.environment.gridmap[tuple(next_)] < 0.8

    def choose_dir(self) -> Tuple[int, int]:
        # we will move by always sticking to the wall on our right and moving forward.
        # if there's no wall on our right, we'll turn right, take a few steps, and try to stick to the wall
        # on our right again.
        # if we cannot move forward (or right), we will turn left.
        dir1, dir2 = self.current_dirs

        # At first go right as far as we can.
        if self.init_moves > 0:
            # TODO: don't always try to go right, choose direction leading to the closest wall...
            if self.can_go_in_direction(Direction.RIGHT.value):
                self.init_moves -= 1
                return Direction.RIGHT.value
            self.init_moves = 0

        if self.turn:
            self.turn_remaining_moves -= 1
            if self.turn_remaining_moves == 0:
                self.turn = False
            return self.current_dirs[0].value

        if not self.can_go_in_direction(dir2.value):
            if self.can_go_in_direction(dir1.value):
                # we can stick to the wall
                return dir1.value
            else:
                # we can't go up or right, we have to turn left
                self.turn = True
                self.turn_remaining_moves = 3
                self.current_dirs = tuple(self.anti_clockwise_turn(dir) for dir in self.current_dirs)
                return self.current_dirs[0].value

        # initiate turn right
        self.turn = True
        self.current_dirs = tuple(self.clockwise_turn(dir) for dir in self.current_dirs)
        self.turn_remaining_moves = 5
        return self.current_dirs[0].value

    def step(self) -> None:
        """
        Mapping agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the occupancy map beliefs/probabilities about their state
            * choosing and executing the next agent action in the environment
        """
        lidar = self.environment.lidar()
        self.occ_map.map_update(self.environment.position(),
                                lidar[0],
                                lidar[1])

        dir = self.choose_dir()

        self.environment.step(dir)

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of given cell occupancy
        """
        return 1 - 1 / (1 + np.exp(self.occ_map.occ_map))


def ray_trace_test():
    # angle = math.pi * 3 / 4  # 135 degrees
    angle = math.pi / 8  # 135 degrees
    distance = 3 * math.sqrt(2)
    pos = (0, 0)
    for cell in OccupancyMap._ray_trace_passed_cells_revised(angle, distance, pos):
        print(cell)


if __name__ == "__main__":
    maze = generate_maze((11, 11))

    env = Environment(
        maze,
        resolution=1 / 11 / 10,
        agent_init_pos=(0.136, 0.136),
        goal_position=(0.87, 0.87),
        lidar_angles=256
    )
    agent = MappingAgent(env)

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
