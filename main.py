import turtle
import random
import time
import math
import pickle
import os
import tkinter as tk
from tkinter import ttk, Canvas

# Configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BOT_RADIUS = 10
OBSTACLE_SIZE = 30
FINISH_SIZE = 40
SIGHT_RANGE = 100
POPULATION_SIZE = 30
GENERATION_TIME = 15  # seconds

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights (Input -> Hidden)
        self.w_ih = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        # Weights (Hidden -> Output)
        self.w_ho = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        
        # Bias
        self.b_h = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b_o = [random.uniform(-1, 1) for _ in range(output_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict(self, inputs):
        # Hidden layer
        hidden = []
        for j in range(self.hidden_size):
            sum_val = self.b_h[j]
            for i in range(self.input_size):
                sum_val += inputs[i] * self.w_ih[i][j]
            hidden.append(self.sigmoid(sum_val))
        
        # Output layer
        outputs = []
        for k in range(self.output_size):
            sum_val = self.b_o[k]
            for j in range(self.hidden_size):
                sum_val += hidden[j] * self.w_ho[j][k]
            outputs.append(self.sigmoid(sum_val)) # 0-1 for speed/turn
            
        return outputs

    def mutate(self, rate):
        def mutate_val(val):
            if random.random() < rate:
                return val + random.gauss(0, 0.2)
            return val

        self.w_ih = [[mutate_val(w) for w in row] for row in self.w_ih]
        self.w_ho = [[mutate_val(w) for w in row] for row in self.w_ho]
        self.b_h = [mutate_val(b) for b in self.b_h]
        self.b_o = [mutate_val(b) for b in self.b_o]

    def copy(self):
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.w_ih = [row[:] for row in self.w_ih]
        new_nn.w_ho = [row[:] for row in self.w_ho]
        new_nn.b_h = self.b_h[:]
        new_nn.b_o = self.b_o[:]
        return new_nn

class Obstacle:
    def __init__(self, x, y, size=30):
        self.x = x
        self.y = y
        self.size = size
        self.color = "red"

    def update(self):
        pass

    def draw(self, pen, view_offset):
        pen.goto(self.x - view_offset[0], self.y - view_offset[1])
        pen.dot(self.size * 2, self.color)

class MovingObstacle(Obstacle):
    def __init__(self, x, y, range_x=100, speed=2, axis='x'):
        super().__init__(x, y, 30)
        self.start_x = x
        self.start_y = y
        self.range = range_x
        self.speed = speed
        self.axis = axis
        self.offset = 0
        self.direction = 1
        self.color = "maroon"

    def update(self):
        self.offset += self.speed * self.direction
        if abs(self.offset) > self.range:
            self.direction *= -1
            
        if self.axis == 'x':
            self.x = self.start_x + self.offset
        else:
            self.y = self.start_y + self.offset

class Predator:
    def __init__(self):
        self.x = 0
        self.y = -SCREEN_HEIGHT/2
        self.speed = 1.5
        self.size = 20
        self.color = "purple"
        self.active = False
    
    def spawn(self):
        self.active = True
        self.x = 0
        self.y = -SCREEN_HEIGHT/2

    def update(self, target_bot):
        if not self.active or not target_bot:
            return
            
        # Chase logic
        dx = target_bot.x - self.x
        dy = target_bot.y - self.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
            
    def draw(self, pen, view_offset):
        if self.active:
            pen.goto(self.x - view_offset[0], self.y - view_offset[1])
            pen.dot(self.size * 2, self.color)


class Checkpoint:
    def __init__(self, x, y, width, height, order):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.order = order
        self.reached = False 

    def contains(self, x, y):
        return (self.x - self.width/2 < x < self.x + self.width/2 and
                self.y - self.height/2 < y < self.y + self.height/2)

class Bot:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 360)
        self.speed = 0
        self.color = (random.random(), random.random(), random.random())
        self.brain = brain if brain else NeuralNetwork(5, 8, 2)
        
        self.alive = True
        self.finished = False
        self.fitness = 0
        self.checkpoint_index = 0

    def update(self, obstacles, finish_line, predator=None, checkpoints=[]):
        if not self.alive or self.finished:
            return

        # Sensors
        inputs = self.get_sensors(obstacles, finish_line)
        outputs = self.brain.predict(inputs)
        
        self.speed = outputs[0] * 5
        turn = (outputs[1] - 0.5) * 20 
        
        self.angle += turn
        
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        
        self.check_collision(obstacles, predator)
        self.check_finish(finish_line)
        self.check_bounds()
        
        if self.alive and not self.finished and checkpoints:
            self.check_checkpoints(checkpoints)

        dist = math.hypot(finish_line[0] - self.x, finish_line[1] - self.y)
        
        self.fitness = self.checkpoint_index * 1000
        
        if self.checkpoint_index < len(checkpoints):
             next_cp = checkpoints[self.checkpoint_index]
             dist_cp = math.hypot(next_cp.x - self.x, next_cp.y - self.y)
             self.fitness += (1000 / (dist_cp + 1))
        else:
             self.fitness += (1000 / (dist + 1))
             
        if self.finished:
            self.fitness += 5000

    def check_checkpoints(self, checkpoints):
         if self.checkpoint_index < len(checkpoints):
             cp = checkpoints[self.checkpoint_index]
             if cp.contains(self.x, self.y):
                 self.checkpoint_index += 1
                 cp.reached = True

    def get_sensors(self, obstacles, finish_line):
        rays = [-30, 0, 30]
        readings = []
        
        for r_angle in rays:
            angle_rad = math.radians(self.angle + r_angle)
            min_dist = SIGHT_RANGE
            
            rx = math.cos(angle_rad)
            ry = math.sin(angle_rad)
            
            for obs in obstacles:
                vx = obs.x - self.x
                vy = obs.y - self.y
                dot = vx * rx + vy * ry
                if dot > 0: 
                    perp_dist = math.hypot(vx - dot*rx, vy - dot*ry)
                    if perp_dist < obs.size:
                        dist = dot - math.sqrt(obs.size**2 - perp_dist**2)
                        if 0 < dist < min_dist:
                            min_dist = dist
                            
            readings.append(min_dist / SIGHT_RANGE) 
            
        dx = finish_line[0] - self.x
        dy = finish_line[1] - self.y
        dist_to_finish = math.hypot(dx, dy)
        angle_to_finish = math.degrees(math.atan2(dy, dx))
        angle_diff = (angle_to_finish - self.angle + 180) % 360 - 180
        
        readings.append(min(dist_to_finish / 800, 1)) 
        readings.append(angle_diff / 180) 
        
        return readings

    def check_collision(self, obstacles, predator):
        for obs in obstacles:
            if math.hypot(obs.x - self.x, obs.y - self.y) < BOT_RADIUS + obs.size:
                self.alive = False
                self.color = (0.5, 0.5, 0.5) 
                return

        if predator and predator.active:
             if math.hypot(predator.x - self.x, predator.y - self.y) < BOT_RADIUS + predator.size:
                self.alive = False
                self.color = (0.5, 0, 0.5) 

    def check_bounds(self):
        if not (-SCREEN_WIDTH/2 < self.x < SCREEN_WIDTH/2 and -SCREEN_HEIGHT/2 < self.y < SCREEN_HEIGHT/2):
            self.alive = False
            self.color = (0.5, 0.5, 0.5)
            
    def check_finish(self, finish_line):
        fx, fy = finish_line
        if abs(self.x - fx) < FINISH_SIZE/2 and abs(self.y - fy) < FINISH_SIZE/2:
            self.finished = True
            self.alive = False 
            self.color = (0, 1, 0) 

    def draw(self, pen, view_offset=(0,0), show_rays=False):
        draw_x = self.x - view_offset[0]
        draw_y = self.y - view_offset[1]
        
        pen.penup()
        pen.goto(draw_x, draw_y)
        pen.setheading(self.angle)
        
        pen.color(self.color)
        pen.shape("triangle")
        pen.shapesize(0.5, 0.5)
        pen.stamp()

        if show_rays:
            rays = [-30, 0, 30]
            for r_angle in rays:
                pen.goto(draw_x, draw_y)
                pen.setheading(self.angle + r_angle)
                pen.pendown()
                pen.pencolor("green") 
                pen.forward(SIGHT_RANGE)
                pen.penup()

class Game:
    def __init__(self, turtle_screen, app):
        self.screen = turtle_screen
        self.app = app
        
        self.pen = turtle.RawTurtle(self.screen)
        self.pen.hideturtle()
        self.pen.penup()
        self.pen.speed(0)
        
        self.population = []
        self.obstacles = []
        self.checkpoints = [] 
        self.finish_line = (0, 0)
        self.generation = 1
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        self.best_bot = None
        self.predator = Predator()
        
        self.camera_follow = False
        self.view_offset = (0, 0)
        self.level_mode = "Random"
        self.is_paused = False
        self.mutation_rate = 0.1

    def start(self):
        self.reset_level()
        self.create_population()

    def add_boundary_walls(self):
        for x in range(int(-SCREEN_WIDTH/2), int(SCREEN_WIDTH/2), 40):
            self.obstacles.append(Obstacle(x, SCREEN_HEIGHT/2 - 20, 20))
        for x in range(int(-SCREEN_WIDTH/2), int(SCREEN_WIDTH/2), 40):
            self.obstacles.append(Obstacle(x, -SCREEN_HEIGHT/2 + 20, 20))
        for y in range(int(-SCREEN_HEIGHT/2), int(SCREEN_HEIGHT/2), 40):
            self.obstacles.append(Obstacle(-SCREEN_WIDTH/2 + 20, y, 20))
        for y in range(int(-SCREEN_HEIGHT/2), int(SCREEN_HEIGHT/2), 40):
            self.obstacles.append(Obstacle(SCREEN_WIDTH/2 - 20, y, 20))

    def generate_maze(self):
         for x in [-200, -100, 0, 100, 200]:
             gap = random.randint(-200, 200)
             for y in range(int(-SCREEN_HEIGHT/2), int(SCREEN_HEIGHT/2), 40):
                 if not (gap - 50 < y < gap + 50):
                      self.obstacles.append(Obstacle(x, y, 30))
         
         self.checkpoints = []
         self.checkpoints.append(Checkpoint(-150, 0, 50, 600, 1))
         self.checkpoints.append(Checkpoint(-50, 0, 50, 600, 2))
         self.checkpoints.append(Checkpoint(50, 0, 50, 600, 3))
         self.checkpoints.append(Checkpoint(150, 0, 50, 600, 4))
         
    def reset_level(self):
        start_x = -SCREEN_WIDTH/2 + 50
        self.finish_line = (SCREEN_WIDTH/2 - 50, 0)
        self.obstacles = []
        self.checkpoints = []
        self.predator.active = False
        
        self.add_boundary_walls()
        
        # Generation schedule
        if self.generation % 20 == 0: self.level_mode = "Maze"
        elif self.generation % 15 == 0: self.level_mode = "Gauntlet"
        elif self.generation % 10 == 0: self.level_mode = "Wall"
        elif self.generation % 5 == 0: self.level_mode = "Predator"
        elif self.generation == 1: self.level_mode = "Random"

        if self.level_mode == "Random" or self.level_mode == "Predator":
             for _ in range(5 + int(self.generation/2)):
                ox = random.randint(-250, 250)
                oy = random.randint(-200, 200)
                if abs(ox - start_x) > 100 and abs(ox - self.finish_line[0]) > 50:
                    self.obstacles.append(Obstacle(ox, oy))
             if self.level_mode == "Predator": self.predator.spawn()

        elif self.level_mode == "Wall":
            gap_y = random.randint(-100, 100)
            wall_x = 0
            for y in range(int(-SCREEN_HEIGHT/2), int(SCREEN_HEIGHT/2), 40):
                if not (gap_y - 60 < y < gap_y + 60):
                    self.obstacles.append(Obstacle(wall_x, y, 30))
        
        elif self.level_mode == "Gauntlet":
            self.obstacles.append(MovingObstacle(0, 0, range_x=0, axis='y', speed=3)) 
            self.obstacles.append(MovingObstacle(-100, 100, range_x=100, axis='x', speed=4))
            self.obstacles.append(MovingObstacle(100, -100, range_x=100, axis='x', speed=4))
            self.obstacles.append(MovingObstacle(200, 0, range_x=0, axis='y', speed=5))
            
        elif self.level_mode == "Maze":
             self.generate_maze()
             
        # Update App
        self.app.update_level_label(self.level_mode)

    def next_level(self):
        modes = ["Random", "Wall", "Gauntlet", "Predator", "Maze"]
        current_idx = modes.index(self.level_mode) if self.level_mode in modes else 0
        self.level_mode = modes[(current_idx + 1) % len(modes)]
        self.generation += 1 
        self.reset_level()
        self.create_population(self.population)

    def handle_click(self, x, y):
        real_x = x + self.view_offset[0]
        real_y = y + self.view_offset[1]
        self.add_obstacle(real_x, real_y)
        print(f"Added obstacle at {real_x:.0f}, {real_y:.0f}")

    def add_obstacle(self, x, y):
        self.obstacles.append(Obstacle(x, y))

    def create_population(self, old_pop=None):
        self.population = []
        start_x = -SCREEN_WIDTH/2 + 50
        
        if old_pop:
            old_pop.sort(key=lambda b: b.fitness, reverse=True)
            self.best_bot = old_pop[0]
            
            # Update Graph
            self.app.add_data_point(self.generation, self.best_bot.fitness)
            
            new_pop = [Bot(start_x, 0, brain=self.best_bot.brain.copy())] 
            
            for _ in range(POPULATION_SIZE - 1):
                parent = random.choice(old_pop[:10]) 
                child_brain = parent.brain.copy()
                child_brain.mutate(self.mutation_rate) # Use dynamic rate
                new_pop.append(Bot(start_x, 0, brain=child_brain))
            
            self.population = new_pop
        else:
            for _ in range(POPULATION_SIZE):
                self.population.append(Bot(start_x, 0))
        
        self.start_time = time.time()
        self.app.update_gen_label(self.generation)

    def update(self):
        if self.is_paused: return

        elapsed = time.time() - self.start_time
        self.last_update_time = time.time()
        
        for obs in self.obstacles:
            obs.update()
            
        closest_bot = None
        min_dist = float('inf')
        alive_count = 0
        current_best = None
        max_dist = -float('inf')
        
        for bot in self.population:
            bot.update(self.obstacles, self.finish_line, self.predator, self.checkpoints)
            if bot.alive and not bot.finished:
                alive_count += 1
                d = math.hypot(bot.x - self.predator.x, bot.y - self.predator.y)
                if d < min_dist:
                    min_dist = d
                    closest_bot = bot
            
            if current_best is None or bot.fitness > current_best.fitness:
                current_best = bot

        if self.predator.active:
            self.predator.update(closest_bot)

        if current_best and self.camera_follow:
             self.view_offset = (current_best.x, current_best.y)
        elif not self.camera_follow:
             self.view_offset = (0, 0)
                
        # Draw (optimized: skip if fast)
        # Note: Turtle inside Tkinter can be slow, so we might want to skip frames
        if not self.app.fast_mode.get() or self.generation % 5 == 0:
            self.draw(current_best)
            
        if alive_count == 0 or elapsed > GENERATION_TIME:
            self.generation += 1
            self.reset_level() 
            self.create_population(self.population)

        # Update stats
        self.app.update_stats(GENERATION_TIME - elapsed, alive_count, self.best_bot.fitness if self.best_bot else 0)

    def draw(self, focus_bot):
        self.pen.clear()
        
        fx, fy = self.finish_line
        self.pen.goto(fx - self.view_offset[0], fy - self.view_offset[1])
        self.pen.dot(FINISH_SIZE, "lime")
        
        for obs in self.obstacles:
            obs.draw(self.pen, self.view_offset)
            
        self.predator.draw(self.pen, self.view_offset)
            
        for bot in self.population:
            is_focus = (bot == focus_bot)
            bot.draw(self.pen, self.view_offset, show_rays=is_focus)
            
        self.screen.update()

    def save_best(self):
        if self.best_bot:
            try:
                with open("best_brain.pkl", "wb") as f:
                    pickle.dump(self.best_bot.brain, f)
                print("Saved best brain!")
            except Exception as e:
                print(e)

    def load_best(self):
        if os.path.exists("best_brain.pkl"):
            try:
                with open("best_brain.pkl", "rb") as f:
                    brain = pickle.load(f)
                    self.population = [Bot(-SCREEN_WIDTH/2 + 50, 0, brain.copy()) for _ in range(POPULATION_SIZE)] 
                    self.start_time = time.time()
                print("Loaded brain!")
            except Exception as e:
                print(e)

class BotRaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bot Race Evolution Pro")
        self.root.geometry("1000x700")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main Layout
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left: Game Canvas
        self.canvas_frame = ttk.Frame(self.main_container, borderwidth=2, relief="sunken")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = Canvas(self.canvas_frame, width=800, height=600, bg="#f0f0f0")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Turtle Screen initialization
        self.screen = turtle.TurtleScreen(self.canvas)
        self.screen.bgcolor("#f0f0f0")
        self.screen.tracer(0)
        
        # Game Instance
        self.game = Game(self.screen, self)
        
        # Canvas Events
        self.canvas.bind("<Button-1>", lambda e: self.game.handle_click(e.x - 400, 300 - e.y)) # Correct coords
        
        # Right: Controls
        self.sidebar = ttk.Frame(self.main_container, width=200, padding=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.create_sidebar_widgets()
        
        # Loop
        self.running = True
        self.game.start()
        self.update_loop()

    def create_sidebar_widgets(self):
        # Stats
        ttk.Label(self.sidebar, text="Statistics", font=("Arial", 14, "bold")).pack(pady=5)
        self.lbl_gen = ttk.Label(self.sidebar, text="Gen: 1", font=("Arial", 12))
        self.lbl_gen.pack(anchor="w")
        self.lbl_level = ttk.Label(self.sidebar, text="Level: Random", font=("Arial", 12))
        self.lbl_level.pack(anchor="w")
        self.lbl_time = ttk.Label(self.sidebar, text="Time: 15.0", font=("Arial", 12))
        self.lbl_time.pack(anchor="w")
        self.lbl_alive = ttk.Label(self.sidebar, text="Alive: 0", font=("Arial", 12))
        self.lbl_alive.pack(anchor="w")
        self.lbl_fit = ttk.Label(self.sidebar, text="Best Fit: 0", font=("Arial", 12))
        self.lbl_fit.pack(anchor="w")
        
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill='x', pady=10)
        
        # Controls
        ttk.Label(self.sidebar, text="Controls", font=("Arial", 14, "bold")).pack(pady=5)
        
        btn_frame = ttk.Frame(self.sidebar)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="Next Level", command=self.game.next_level).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Camera Toggle", command=self.toggle_camera).pack(fill='x', pady=2)
        
        # Settings
        ttk.Label(self.sidebar, text="Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.fast_mode = tk.BooleanVar()
        ttk.Checkbutton(self.sidebar, text="Fast Mode (No Draw)", variable=self.fast_mode).pack(anchor="w")
        
        ttk.Label(self.sidebar, text="Mutation Rate").pack(anchor="w", pady=(5,0))
        self.scale_mut = tk.Scale(self.sidebar, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, command=self.update_mutation)
        self.scale_mut.set(0.1)
        self.scale_mut.pack(fill='x')
        
        ttk.Label(self.sidebar, text="Sim Speed (ms delay)").pack(anchor="w", pady=(5,0))
        self.speed_delay = tk.IntVar(value=10)
        self.scale_speed = tk.Scale(self.sidebar, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.speed_delay)
        self.scale_speed.pack(fill='x')

        ttk.Separator(self.sidebar, orient='horizontal').pack(fill='x', pady=10)
        
        # Save/Load
        btn_io_frame = ttk.Frame(self.sidebar)
        btn_io_frame.pack(fill='x')
        ttk.Button(btn_io_frame, text="Save Brain", command=self.game.save_best).pack(side=tk.LEFT, expand=True, padx=1)
        ttk.Button(btn_io_frame, text="Load Brain", command=self.game.load_best).pack(side=tk.LEFT, expand=True, padx=1)

        # Graph Canvas
        ttk.Label(self.sidebar, text="Fitness History", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        self.graph_canvas = Canvas(self.sidebar, width=180, height=100, bg="white", highlightthickness=1, highlightbackground="gray")
        self.graph_canvas.pack()
        self.fitness_history = []

    def update_mutation(self, val):
        self.game.mutation_rate = float(val)

    def toggle_camera(self):
        self.game.camera_follow = not self.game.camera_follow

    def update_stats(self, time_left, alive, fitness):
        self.lbl_time.config(text=f"Time: {max(0, time_left):.1f}")
        self.lbl_alive.config(text=f"Alive: {alive}")
        self.lbl_fit.config(text=f"Best Fit: {fitness:.0f}")

    def update_gen_label(self, gen):
        self.lbl_gen.config(text=f"Gen: {gen}")

    def update_level_label(self, level):
        self.lbl_level.config(text=f"Level: {level}")

    def add_data_point(self, gen, fitness):
        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 20:
            self.fitness_history.pop(0)
        self.draw_graph()

    def draw_graph(self):
        self.graph_canvas.delete("all")
        if not self.fitness_history: return
        
        w = 180
        h = 100
        max_val = max(self.fitness_history) if max(self.fitness_history) > 0 else 1
        
        # Scaling
        points = []
        num_pts = len(self.fitness_history)
        step_x = w / (num_pts - 1) if num_pts > 1 else w
        
        for i, val in enumerate(self.fitness_history):
            x = i * step_x
            y = h - (val / max_val * h)
            points.append(x)
            points.append(y)
            
        if len(points) >= 4:
            self.graph_canvas.create_line(points, fill="blue", width=2)

    def update_loop(self):
        if self.running:
            self.game.update()
            # Schedule next update
            delay = self.speed_delay.get()
            # If Fast Mode, use minimal delay
            if self.fast_mode.get(): delay = 1
            self.root.after(delay, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = BotRaceApp(root)
    root.mainloop()
