import matplotlib.pyplot as plt
import numpy as np

from collections import deque

import heapq


def getAction(x, y, new_x, new_y):
        """
            Action Space :  0 ,     1,      2,     3
            Actions : Move  Right, Left,    Up,   Down
            Moves :         (0,1), (0,-1) (-1,0) (1,0)
        """
        if new_x == x + 1:
            return 3
        elif new_x == x - 1:
            return 2
        elif new_y == y + 1:
            return 0
        elif new_y == y - 1:
            return 1
        else:
            # Handle cases where the new position is not adjacent to the current position
            return None
        
def heuristic(node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

class DFSAgent:
    
    def __init__(self):
        self.visited = set()
        self.fringe = []
        
    def dfs_search(self, map, start, goal):
        print("Depth First Search (DFS)")
        rows = len(map)
        cols = len(map[0])

        startRow, startCol = start
        self.fringe = [(startRow, startCol, [], [])]

        def is_movable(row, col):
            return 0<=row<rows and 0<=col<cols and map[row][col]!='#'
        
        while self.fringe:
            row, col, path, actions = self.fringe.pop()
            # print("Path: ", path)
            
            if (row, col) == goal:
                return True, path + [(row, col)], actions
            
            self.visited.add((row, col))
            moves = [(0, -1),(1, 0),(0, 1), (-1, 0)]
            for (dr, dc) in moves:
                nextRow, nextCol = row + dr, col + dc
                if is_movable(nextRow, nextCol) and (nextRow, nextCol) not in self.visited:
                    nextPath = path + [(row, col)]
                    nextActions = actions + [getAction(row, col, nextRow, nextCol)]
                    self.fringe.append((nextRow, nextCol, nextPath, nextActions))
        return False, [], []
    
class BFSAgent:
    def __init__(self):
            self.visited = set()
            self.fringe = []
    def bfs_search(self, map, start, goal):
        print("Breadth First Search  - BFS ")
        rows = len(map)
        cols = len(map[0])
        def is_movable(row, col):
            return 0<=row<rows and 0<=col<cols and map[row][col]!='#'
        
        self.fringe = deque([(start[0], start[1], [], [])])

        while self.fringe:
            row, col, path, actions = self.fringe.popleft()
            if (row, col) == goal:
                return True, path+[(row, col)], actions
            
            self.visited.add((row, col))

            moves = [(0, -1),(1, 0), (0, 1), (-1, 0)]
            for (dr, dc) in moves:
                nextRow, nextCol = row+dr, col+dc
                if is_movable(nextRow, nextCol) and (nextRow,nextCol) not in self.visited:
                    nextActions = actions + [getAction(row, col, nextRow, nextCol)]
                    self.fringe.append((nextRow,nextCol, path+[(nextRow,nextCol)], nextActions))
        
        return False, [],[]

class UCSAgent:
    def __init__(self):
        self.visited = set()
        self.fringe = []
        self.move_costs = [[0, 0, 0, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 1, 1, 0, 1, 1],
                          [1, 57, 57, 58, 55, 56, 57],
                          [1, 1, 1, 1, 2, 1, 1]]

    def ucs_search(self, map, start, goal):
        print("Unform cost search - UCS")
        rows = len(map)
        cols = len(map[0])

        def is_movable(row, col):
            return 0 <= row < rows and 0 <= col < cols and map[row][col] != '#'

        startRow, startCol = start
        self.fringe = [(0, startRow, startCol, [], [])]  # Tuple format: (cost, row, col, path, action)

        while self.fringe:
            cost, row, col, path, actions = heapq.heappop(self.fringe)
            
            if (row, col) == goal:
                return True, path + [(row, col)], actions

            self.visited.add((row, col))
            moves = [(0, -1),(1, 0), (0, 1), (-1, 0)]
            
            for dr, dc in (moves):
                nextRow, nextCol = row + dr, col + dc
                if is_movable(nextRow, nextCol) and (nextRow, nextCol) not in self.visited:
                    new_cost = cost + self.move_costs[row][col]  # Use move-specific cost i.e row, col
                    heapq.heappush(self.fringe, (new_cost, nextRow, nextCol, path + [(nextRow, nextCol)], actions + [getAction(row, col, nextRow, nextCol)]))

        return False, [],[]
    
class AStarAgent:

    def __init__(self):
        self.visited = set()
        self.fringe = []
        self.cost_matrix = [[0, 0, 0, 0, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 1, 1, 0, 1, 1],
                            [1, 57, 57, 58, 55, 56, 57],
                            [1, 1, 1, 1, 2, 1, 1]]

    def astar_search(self, map, start, goal):
        rows = len(map)
        cols = len(map[0])

        def is_movable(row, col):
            return 0 <= row < rows and 0 <= col < cols and map[row][col] != '#'

        startRow, startCol = start
        self.fringe = [(heuristic(start, goal), 0, startRow, startCol, [], [])]
        cost_so_far = {}
        visit_count = np.zeros((rows, cols), dtype=int)

        while self.fringe:
            _, cost, row, col, path, actions = heapq.heappop(self.fringe)
            if (row, col) == goal:
                return True, path + [(row, col)], visit_count, actions

            self.visited.add((row, col))

            moves = [(0, -1),(1, 0), (0, 1), (-1, 0)]

            for dr, dc in moves:
                nextRow, nextCol = row + dr, col + dc
                new_cost = cost + self.cost_matrix[row][col]  # Assuming uniform cost for each move

                if is_movable(nextRow, nextCol) and (nextRow, nextCol) not in self.visited:
                    if (nextRow, nextCol) not in cost_so_far or new_cost < cost_so_far[(nextRow, nextCol)]:
                        heapq.heappush(self.fringe, (new_cost + heuristic((nextRow, nextCol), goal), new_cost, nextRow, nextCol, path + [(nextRow, nextCol)], actions + [getAction(row, col, nextRow, nextCol)]))
                        cost_so_far[(nextRow, nextCol)] = new_cost
                        visit_count[nextRow][nextCol] = 1

        return False, [], visit_count, []


"""
Testing the path to see if executed correctly  
"""                    
# map = [
#     ['.', '.', '.', '#', '.', '.', '.'],
#     ['.', '#', '.', '#', '.', '.', '.'],
#     ['.', '.', '.', '.', '.', '.', '.'],
#     ['.', '#', '#', '#', '.', '.', '#'],
#     ['.', '.', '.', 'G', '#', '.', '.'],
#     ['S', '#', '.', '.', '.', '.', '.'],
# ]

# start = (5, 0)
# goal = (4,3)

# agent = AStarAgent()
# isFound, path, visit_count, action = agent.astar_search(map, start, goal)
# agent = UCSAgent()
# isFound, path, action = agent.ucs_search(map, start, goal)
# agent = BFSAgent()
# isFound, path, action = agent.bfs_search(map, start, goal)
# agent = DFSAgent()
# isFound, path, action = agent.dfs_search(map, start, goal)

# print('Path', path)
# print('Action', action)
# if isFound:
#     print("Path found")
#     path_map = [row[:] for row in map]
#     for (row, col) in path:
#         if path_map[row][col]!='G':
#             path_map[row][col] = '+'
#     for row in path_map:
#         print(" ".join(row))
    
# else:
#     print("Path Not Found")
# print(agent.dfs_search(map, start, goal))


# Plot the map and path

# if isFound:
#     path_array = np.zeros((len(map), len(map[0])), dtype=int)
#     for i, row in enumerate(map):
#         for j, cell in enumerate(row):
#             if cell == '#':
#                 path_array[i][j] = 1

#     for (row, col) in path:
#         path_array[row][col] = 2  # Use a different value for the path
#     path_array[start[0]][start[1]] = 3  # Use a different value for the start
#     path_array[goal[0]][goal[1]] = 4  # Use a different value for the goal

#     cmap = plt.matplotlib.colors.ListedColormap(['white', 'black', 'blue', 'green', 'red'])
#     plt.imshow(path_array, cmap=cmap)

# plt.title('DFS Path')
# plt.title('BFS Path')
# plt.title('UCS Path')
# plt.title('AStar Path')
# plt.show()
# print(path)




