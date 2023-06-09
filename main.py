import turtle
import random
import time
import threading
import math

# Set up the screen
screen = turtle.Screen()
screen.setup(600, 600)
screen.title("Bot Race")
screen.bgcolor("white")

# Create the turtle for drawing game objects
drawer = turtle.Turtle()
drawer.hideturtle()

# Bot properties
bot_radius = 10

# Obstacle properties
obstacle_size = 20

# Finish line properties
finish_line_width = 40
finish_line_height = 20

# Neural network properties
input_size = 2  # X and Y positions
output_size = 2  # Movement in X and Y directions
num_synapses = 100
num_hidden_layers = 1

# Population properties
population_size = 50
mutation_rate = 0.1

# Bounded area properties
area_min = -200
area_max = 200

# Maximum time limit
time_limit = 10  # seconds

# Initialize bots
def initialize_bots():
    bots = []
    for _ in range(population_size):
        bot = {
            'x': random.randint(area_min + bot_radius, area_max - bot_radius),
            'y': random.randint(area_min + bot_radius, area_max - bot_radius),
            'dx': 0,
            'dy': 0,
            'fitness': 0,
            'brain': initialize_neural_network()
        }
        bots.append(bot)
    return bots

# Function to initialize the neural network
def initialize_neural_network():
    network = []
    # Input to hidden layer
    for _ in range(num_synapses):
        synapse = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        network.append(synapse)
    # Hidden layer to output
    for _ in range(num_synapses):
        synapse = [random.uniform(-1, 1) for _ in range(num_hidden_layers + 1)]
        network.append(synapse)
    return network

# Function to draw the game objects on the screen
def draw_game_objects():
    drawer.clear()

    drawer.penup()
    drawer.goto(finish_line['x'], finish_line['y'])
    drawer.pendown()
    drawer.fillcolor('green')
    drawer.begin_fill()
    for _ in range(2):
        drawer.forward(finish_line_width)
        drawer.left(90)
        drawer.forward(finish_line_height)
        drawer.left(90)
    drawer.end_fill()

    for obstacle in obstacles:
        drawer.penup()
        drawer.goto(obstacle['x'], obstacle['y'])
        drawer.pendown()
        drawer.fillcolor(obstacle['color'])
        drawer.begin_fill()
        for _ in range(4):
            drawer.forward(obstacle_size)
            drawer.left(90)
        drawer.end_fill()

    drawer.penup()
    for bot in bots:
        drawer.goto(bot['x'], bot['y'])
        drawer.pendown()
        drawer.fillcolor('red')
        drawer.begin_fill()
        drawer.circle(bot_radius)
        drawer.end_fill()

    screen.update()

# Function to perform a bot move based on its brain
def move_bot(bot):
    inputs = [bot['x'], bot['y']]
    for synapse in bot['brain']:
        inputs.append(activation_function(dot_product(inputs, synapse)))

    dx = inputs[-2]
    dy = inputs[-1]

    angle = math.atan2(dy, dx)  # Calculate the angle based on movement direction

    # Rotate the bot randomly within a certain range to allow for exploration
    rotation_angle = random.uniform(-math.pi / 4, math.pi / 4)
    angle += rotation_angle

    speed = math.sqrt(dx ** 2 + dy ** 2)  # Calculate the speed based on movement direction

    new_x = bot['x'] + math.cos(angle) * speed
    new_y = bot['y'] + math.sin(angle) * speed

    # Check collision with obstacles
    for obstacle in obstacles:
        if is_collision(new_x, new_y, bot_radius, obstacle['x'], obstacle['y'], obstacle_size):
            # Collision detected, rotate bot randomly and move away from the obstacle
            angle += random.uniform(-math.pi / 4, math.pi / 4) + math.pi
            new_x = bot['x'] + math.cos(angle) * speed
            new_y = bot['y'] + math.sin(angle) * speed
            break

    # Check collision with area boundaries
    if new_x < area_min + bot_radius:
        new_x = area_min + bot_radius
    elif new_x > area_max - bot_radius:
        new_x = area_max - bot_radius

    if new_y < area_min + bot_radius:
        new_y = area_min + bot_radius
    elif new_y > area_max - bot_radius:
        new_y = area_max - bot_radius

    bot['x'] = new_x
    bot['y'] = new_y

# Function to check if two objects collide
def is_collision(x1, y1, size1, x2, y2, size2):
    if (
        x1 + size1 >= x2 and
        x1 <= x2 + size2 and
        y1 + size1 >= y2 and
        y1 <= y2 + size2
    ):
        return True
    return False

# Activation function
def activation_function(x):
    return 1 / (1 + pow(2.71828, -x))

# Dot product
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

# Function to check if a bot reached the finish line
def check_finish(bot):
    if (
        bot['x'] >= finish_line['x'] and
        bot['x'] <= finish_line['x'] + finish_line_width and
        bot['y'] >= finish_line['y'] and
        bot['y'] <= finish_line['y'] + finish_line_height
    ):
        return True
    return False

# Function to update the brains of the bots based on fitness scores
def update_brains():
    bots.sort(key=lambda x: x['fitness'], reverse=True)
    best_bots = bots[:population_size // 2]  # Select the best performing bots

    # Update the bots based on fitness scores
    for i in range(population_size // 2, population_size):
        parent1 = random.choice(best_bots)
        parent2 = random.choice(best_bots)

        new_brain = []
        for j in range(num_synapses):
            if random.random() < mutation_rate:
                synapse = [random.uniform(-1, 1) for _ in range(len(parent1['brain'][j]))]
            else:
                if random.random() < 0.5:
                    synapse = parent1['brain'][j]
                else:
                    synapse = parent2['brain'][j]
            new_brain.append(synapse)

        bots[i]['brain'] = new_brain

# Function to reset the game for the next iteration
def reset_game():
    global bots, finish_line, obstacles
    bots.sort(key=lambda x: x['fitness'], reverse=True)
    best_bots = bots[:population_size // 2]  # Select the best performing bots

    # Update the bots based on fitness scores
    for i in range(population_size // 2, population_size):
        parent1 = random.choice(best_bots)
        parent2 = random.choice(best_bots)

        new_brain = []
        for j in range(num_synapses):
            if random.random() < mutation_rate:
                synapse = [random.uniform(-1, 1) for _ in range(len(parent1['brain'][j]))]
            else:
                if random.random() < 0.5:
                    synapse = parent1['brain'][j]
                else:
                    synapse = parent2['brain'][j]
            new_brain.append(synapse)

        bots[i]['brain'] = new_brain

    # Reset the game parameters
    obstacles = []
    for _ in range(population_size):
        obstacle = {
            'x': random.randint(area_min, area_max - obstacle_size),
            'y': random.randint(area_min, area_max - obstacle_size),
            'color': random.choice(['red', 'green'])
        }
        # Check if obstacle overlaps with the finish line
        while is_collision(obstacle['x'], obstacle['y'], obstacle_size, finish_line['x'], finish_line['y'], finish_line_width):
            obstacle['x'] = random.randint(area_min, area_max - obstacle_size)
            obstacle['y'] = random.randint(area_min, area_max - obstacle_size)
        obstacles.append(obstacle)

    finish_line = {
        'x': random.randint(area_min, area_max - finish_line_width),
        'y': random.randint(area_min, area_max - finish_line_height)
    }

# Initialize turtle properties
drawer.speed(0)
drawer.up()

# Hide the turtle window for the finish line and obstacles
turtle.screensize(10, 10)

# Create the main turtle window
main_window = turtle.Screen()
main_window.title("Bot Race")
main_window.setup(600, 600)
main_window.tracer(0)

running = True
generation = 1

bots = initialize_bots()
finish_line = {
    'x': random.randint(area_min, area_max - finish_line_width),
    'y': random.randint(area_min, area_max - finish_line_height)
}
obstacles = []
for _ in range(population_size):
    obstacle = {
        'x': random.randint(area_min, area_max - obstacle_size),
        'y': random.randint(area_min, area_max - obstacle_size),
        'color': random.choice(['red', 'green'])
    }
    # Check if obstacle overlaps with the finish line
    while is_collision(obstacle['x'], obstacle['y'], obstacle_size, finish_line['x'], finish_line['y'], finish_line_width):
        obstacle['x'] = random.randint(area_min, area_max - obstacle_size)
        obstacle['y'] = random.randint(area_min, area_max - obstacle_size)
    obstacles.append(obstacle)

iteration_start_time = time.time()

while running:
    threads = []
    for bot in bots:
        thread = threading.Thread(target=move_bot, args=(bot,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    draw_game_objects()
    time.sleep(0.001)  # Delay for smoother animation

    # Check if any bot reached the finish line
    if any(check_finish(bot) for bot in bots):
        print("Loading...")
        reset_game()
        generation += 1
        print(f"Generation {generation}")

        # Print the fitness scores
        fitness_scores = [bot['fitness'] for bot in bots]
        print(f"Fitness Scores: {fitness_scores}")

        iteration_start_time = time.time()  # Reset the iteration start time

    # Check if all bots are stuck and exceeded the time limit
    if all(bot['x'] == bots[0]['x'] and bot['y'] == bots[0]['y'] for bot in bots) and time.time() - iteration_start_time > time_limit:
        print("All bots are stuck. Starting a new iteration...")
        reset_game()

    main_window.update()

turtle.done()
