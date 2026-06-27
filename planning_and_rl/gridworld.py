import numpy as np
import copy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pygame
from gym import spaces

from typing import Tuple, Dict, Optional, Iterable


class Environment(object):
    def __init__(self):
        # Initialize elements of the task environment
        self.GH = 5  # Height of gridworld
        self.GW = 5  # Width of gridworld
        self.AGENT = 240.0  # Pixel values of agents
        self.TARGETS = [150.0]  # Pixel values of targets
        self.AGENTS_X = 0  # x-coordinates of initial locations of agents
        self.AGENTS_Y = 0  # y-coordinates of initial locations of agents
        self.OBSTACLE = 100.0
        self.OBSTACLES_YX = [(self.GH - 2, 1), (self.GH - 3, 2), (self.GH - 1, 0), (0, self.GW-2)]
        # self.TARGETS_YX = [(1, self.GW - 1)]  # Locations of the target
        self.TARGETS_YX = [(self.GH - 1, 2)]  # Locations of the target
        self.REWARDS = [1]
        self.ACTIONS = 4  # Move up, down, left, right
        self.action_space = spaces.Discrete(n=4)
        self.action_space.action_meanings = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: "LEFT"}
        self.observation_space = spaces.MultiDiscrete([self.GH, self.GW])
        self.screen = None

        
        # # Set up display
        self.cell_size = 50  # Size of each grid cell
        self.window_width = self.GW * self.cell_size
        self.window_height = self.GH * self.cell_size
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Gridworld Environment')

        # Define colors
        self.agent_color = (0, 255, 0)  # Green
        self.target_colors = [(255, 0, 0)]  # Red
        self.obstacle_color = (128, 128, 128)  # Gray
        self.grid_color = (0, 0, 0)  # Black

    def reset(self):
        '''
        Reset the environment.
        '''
        self.s_t = np.zeros([self.GH, self.GW], dtype=np.float64)
        self.agents_x = copy.deepcopy(self.AGENTS_X)
        self.agents_y = copy.deepcopy(self.AGENTS_Y)
        self.agent_status = False
        self.s_t[self.agents_y][self.agents_x] += self.AGENT
        for l, p in zip(self.TARGETS_YX, self.TARGETS):
            self.s_t[l[0]][l[1]] = p
        for yx in self.OBSTACLES_YX:
            self.s_t[yx[0]][yx[1]] = self.OBSTACLE
        # return self.s_t
        return (self.agents_y, self.agents_x)

    def step(self, action):
        '''
        Change environment state based on actions.
        :param action: List of integers providing actions for each agent
        '''
        dx, dy = self.getDelta(action)
        targetx = self.agents_x + dx
        targety = self.agents_y + dy
        reward = 0 # reward = -0.1
        terminal = False
        info = None
        if self.noCollision(targetx, targety):
            for a, r in zip(self.TARGETS, self.REWARDS):
                if self.s_t[targety][targetx] == a:
                    reward = r
                    terminal = True
                    break
            self.s_t[self.agents_y][self.agents_x] -= self.AGENT
            self.s_t[targety][targetx] += self.AGENT
            self.agents_x = targetx
            self.agents_y = targety
            info = tuple((self.agents_x, self.agents_y))
        # return self.s_t, reward, terminal, info
        return (self.agents_y, self.agents_x), reward, terminal, info

    def simulate_step(self, state: Tuple[int, int], action):
        '''
        Simulate (without taking) a step in the environment.
        '''
        next_state = self._get_next_state(state, action)
        reward = 0 #-0.1
        terminal = False
        info = {}
        for a, r in zip(self.TARGETS, self.REWARDS):
            # if self.s_t[next_state[0]][next_state[1]] == a:
            if self.s_t[state[0]][state[1]] == a:
                reward = r
                terminal = True
                break
            
        return next_state, reward, terminal, info
    
    def getDelta(self, action):
        '''
        Determine the direction that the agent should take based on the action selected.
        :param action: int
        '''
        if action == 0:
            return 0, -1
        elif action == 1:
            return 1, 0
        elif action == 2:
            return 0, 1
        elif action == 3:
            return -1, 0
        elif action == 4:
            return 0, 0

    def noCollision(self, x, y):
        '''
        Check if the x, y coordinate is currently empty.
        :param x: Int, x coordinate
        :param y: Int, y coordinate
        '''
        if (
            x < 0
            or x >= self.GW
            or y < 0
            or y >= self.GH
            or self.s_t[y][x] == self.OBSTACLE
        ):
            return False
        else:
            return True

    def showGrid(self, show_agent=True):
        self.palette = ListedColormap(["w", "gray", "r"])
        walls = np.zeros([self.GH, self.GW])
        for (i,j) in self.OBSTACLES_YX:
            print(i,j)
            walls[i, j] = 1

        num_target = len(self.TARGETS_YX)
        targets = np.zeros([num_target, self.GH, self.GW])
        for i in range(num_target):
            targets[i, self.TARGETS_YX[i][0], self.TARGETS_YX[i][1]] = 1

        image_mat = copy.deepcopy(walls)
        for i in range(num_target):
            image_mat += (i + 2) * targets[i, :, :]
        
        self.figure, self.ax = plt.subplots()
        if show_agent:
            self.ax.plot(self.agents_x, self.agents_y, 'og', markersize=10)
        self.ax.matshow(image_mat, cmap=self.palette)
        
        return self.figure, self.ax


    def render(self):
        """
        Render the current state of the gridworld using Pygame.
        """
        # Initialize a surface with a white background
        surf = pygame.Surface((self.GW * self.cell_size, self.GH * self.cell_size))
        surf.fill((255, 255, 255))

        for y in range(self.GH):
            for x in range(self.GW):
                cell_value = self.s_t[y][x]
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

                if cell_value in self.TARGETS:
                    reward_idx = self.TARGETS.index(cell_value)
                    pygame.draw.rect(surf, self.target_colors[reward_idx], rect)
                elif cell_value == self.OBSTACLE:
                    pygame.draw.rect(surf, self.obstacle_color, rect)
                else:
                    # Draw grid lines for empty cells
                    pygame.draw.rect(surf, self.grid_color, rect, 1)

                if cell_value == self.AGENT:
                    # Calculate the center of the cell
                    center_x = x * self.cell_size + self.cell_size // 2
                    center_y = y * self.cell_size + self.cell_size // 2
                    # Calculate the radius of the circle (adjust as needed)
                    radius = self.cell_size // 2 - 2  # Subtract 2 for padding
                    # Draw the agent as a circle at the center of the cell
                    pygame.draw.circle(surf, self.agent_color, (center_x, center_y), radius)

        self.screen.blit(surf, (0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        """
        Clean up resources before shutting down the environment.

        Returns: None.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
    
    def isValidState(self, x, y):
        '''
        Check if the x, y coordinate is currently empty.
        :param x: Int, x coordinate
        :param y: Int, y coordinate
        '''
        if (
            x < 0
            or x >= self.GH
            or y < 0
            or y >= self.GW
            or self.s_t[x][y] == self.OBSTACLE
        ):
            return False
        else:
            return True
        
    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Gets the next state after the agent performs action 'a' in state 's'. If there is a
        wall in the way, the next state will be the same as the current.

        Args:
            state: current state (before taking the action).
            action: move performed by the agent.

        Returns: a State instance representing the new state.
        """
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0], state[1] + 1)
        elif action == 2:
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            next_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Action value not supported:", action)
        if self.isValidState(next_state[0], next_state[1]):
            return next_state
        return state
    