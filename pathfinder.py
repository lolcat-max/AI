import numpy as np
import math
import tkinter as tk
from tkinter import ttk
import random
from collections import deque
from heapq import heappush, heappop
import threading
import time

# ==========================================
# 1. FAST SOLVERS (Preserved)
# ==========================================

class FastSubsetSum:
    def __init__(self): pass
    
    def solve(self, numbers, target):
        queue = deque([(0, [])])
        visited = set([0])
        
        while queue:
            curr_sum, subset = queue.popleft()
            if curr_sum == target:
                return sorted(subset)
            
            for num in numbers:
                new_sum = curr_sum + num
                if new_sum <= target and new_sum not in visited:
                    visited.add(new_sum)
                    queue.append((new_sum, subset + [num]))
        return None

# ==========================================
# 2. FIXED TKINTER GRID PATH VISUALIZER
# ==========================================

class PathfindingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Physics Pathfinding + Subset Sum Solver (FIXED)")
        self.root.geometry("1000x700")
        
        # Solver
        self.subset_solver = FastSubsetSum()
        
        # Grid state
        self.grid_size = 25
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.grid[5:20, 5:10] = 1  # Walls
        self.grid[12:15, 12:] = 1  # Walls
        
        # Path state - FIXED: Initialize properly
        self.path = []
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([self.grid_size-1.0, self.grid_size-1.0])
        self.is_solving = False  # Pathfinding lock
        
        self.setup_ui()
        self.draw_grid()
        
    def setup_ui(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Solve Path", 
                  command=self.solve_path).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Random Maze", 
                  command=self.random_maze).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear Walls", 
                  command=self.clear_walls).pack(side='left', padx=5)
        
        # Subset Sum frame
        ss_frame = ttk.LabelFrame(self.root, text="Subset Sum Solver")
        ss_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(ss_frame, text="Solve Demo", 
                  command=self.solve_subset_demo).pack(side='left', padx=5)
        
        self.ss_status = ttk.Label(ss_frame, text="Ready")
        self.ss_status.pack(side='left', padx=10)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg='black')
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Info
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill='x', padx=10)
        
        self.info_label = ttk.Label(info_frame, text="Click to add/remove walls (gray=wall)")
        self.info_label.pack()
        
    def cell_size(self):
        return 600 // self.grid_size
    
    def draw_grid(self):
        self.canvas.delete("all")
        cell_size = self.cell_size()
        
        # Draw cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1, y1 = i*cell_size, j*cell_size
                x2, y2 = x1+cell_size, y1+cell_size
                
                color = 'gray20' if self.grid[i,j] else 'gray40'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black', width=1)
        
        # Draw agent (green circle) - FIXED: Safe indexing
        ax = int(np.clip(self.agent_pos[0], 0, self.grid_size-1))
        ay = int(np.clip(self.agent_pos[1], 0, self.grid_size-1))
        self.draw_agent(ax, ay, 'lime', 0.8*cell_size)
        
        # Draw target (red square)
        tx = int(np.clip(self.target_pos[0], 0, self.grid_size-1))
        ty = int(np.clip(self.target_pos[1], 0, self.grid_size-1))
        self.draw_agent(tx, ty, 'red', 0.7*cell_size)
        
        # Draw path - FIXED: Bounds checking
        if self.path and len(self.path) > 1:
            for k in range(min(len(self.path)-1, 100)):  # Limit path segments
                x1, y1 = self.path[k]
                x2, y2 = self.path[k+1]
                self.canvas.create_line(
                    x1*cell_size+cell_size//2, y1*cell_size+cell_size//2,
                    x2*cell_size+cell_size//2, y2*cell_size+cell_size//2,
                    fill='cyan', width=4
                )
    
    def draw_agent(self, x, y, color, size):
        cell_size = self.cell_size()
        self.canvas.create_oval(
            x*cell_size + cell_size//4, y*cell_size + cell_size//4,
            (x+1)*cell_size - cell_size//4, (y+1)*cell_size - cell_size//4,
            fill=color, outline='white', width=2
        )
    
    def on_click(self, event):
        cell_size = self.cell_size()
        col = event.x // cell_size
        row = event.y // cell_size
        
        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            # Don't toggle start/end
            if (col, row) != tuple(self.agent_pos.astype(int)) and (col, row) != tuple(self.target_pos.astype(int)):
                self.grid[col, row] = 1 - self.grid[col, row]
            self.draw_grid()
    
    def is_valid(self, x, y):
        ix, iy = int(np.clip(x, 0, self.grid_size-1)), int(np.clip(y, 0, self.grid_size-1))
        return 0 <= ix < self.grid_size and 0 <= iy < self.grid_size and not self.grid[ix, iy]
    
    def solve_path_astar(self):
        """Fast A* pathfinding - FIXED threading safety"""
        start = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        goal = (int(self.target_pos[0]), int(self.target_pos[1]))
        
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        
        queue = [(0, start)]
        came_from = {start: None}
        g_score = {start: 0}
        visited = set()
        
        while queue:
            _, current = heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                    if current is None:
                        return None
                path.append(start)
                path.reverse()
                return path
            
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if self.is_valid(*neighbor):
                    tent_g = g_score[current] + 1
                    if neighbor not in g_score or tent_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tent_g
                        f_score = tent_g + heuristic(neighbor, goal)
                        heappush(queue, (f_score, neighbor))
        
        return None
    
    def solve_path(self):
        """Solve and animate path - FULLY FIXED"""
        if self.is_solving:
            return
        
        self.is_solving = True
        self.info_label.config(text="Solving...")
        self.root.update()
        
        def animate_path():
            try:
                # Compute path
                new_path = self.solve_path_astar()
                
                # Update on main thread
                self.root.after(0, lambda: self.path_complete(new_path))
                
            except Exception as e:
                self.root.after(0, lambda: self.path_error(str(e)))
        
        threading.Thread(target=animate_path, daemon=True).start()
    
    def path_complete(self, new_path):
        """Handle successful path - main thread only"""
        self.path = new_path or []
        self.is_solving = False
        
        if self.path:
            self.agent_pos = np.array(self.path[0])
            path_length = len(self.path) - 1
            self.info_label.config(text=f"Path found! Length: {path_length}")
            
            # Animate movement
            def step_animation(i=0):
                if i < len(self.path):
                    self.agent_pos = np.array(self.path[i])
                    self.draw_grid()
                    self.root.after(100, lambda: step_animation(i+1))
            
            step_animation()
        else:
            self.info_label.config(text="No path found!")
            self.draw_grid()
    
    def path_error(self, error_msg):
        """Handle path error"""
        self.is_solving = False
        self.info_label.config(text=f"Error: {error_msg}")
        self.draw_grid()
    
    def random_maze(self):
        self.grid = np.random.choice([0,1], size=(self.grid_size, self.grid_size), p=[0.75, 0.25])
        self.grid[0,0] = self.grid[self.grid_size-1,self.grid_size-1] = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([self.grid_size-1.0, self.grid_size-1.0])
        self.path = []
        self.draw_grid()
    
    def clear_walls(self):
        self.grid.fill(0)
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([self.grid_size-1.0, self.grid_size-1.0])
        self.path = []
        self.draw_grid()
    
    def solve_subset_demo(self):
        def run_demo():
            numbers = list(range(1, 51))
            target = 127
            subset = self.subset_solver.solve(numbers, target)
            
            self.root.after(0, lambda: self.ss_status.config(
                text=f"Demo: {subset} (sum={sum(subset) if subset else 'None'})"
            ))
        
        threading.Thread(target=run_demo, daemon=True).start()

# ==========================================
# 3. LAUNCH
# ==========================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingGUI(root)
    root.mainloop()
