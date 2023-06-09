import turtle
import random
import time
import multiprocessing

# Set up the screen
screen = turtle.Screen()
screen.setup(400, 400)
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

# Initialize bots
def initialize_bots():
    bots = []
    for _ in range(population_size):
        bot = {
            'x': random.randint(-180, 180),
            'y': random.randint(-180, 180),
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

    bot['x'] += dx
    bot['y'] += dy

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
    bots.sort(key=lambda x: x['fitness'])

    # Pick the best performing bots and keep their brains intact
    best_bots = bots[population_size // 2:]

    for i in range(population_size // 2, population_size):
        # Select two random parents
        parent1 = random.choice(best_bots)
        parent2 = random.choice(best_bots)

        # Create a new brain based on the parents' brains with some crossover and mutation
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
    bots = initialize_bots()
    finish_line = {
        'x': random.randint(-180, 180 - finish_line_width),
        'y': random.randint(-180, 180 - finish_line_height)
    }
    obstacles = []
    for _ in range(population_size):
        obstacle = {
            'x': random.randint(-180, 180 - obstacle_size),
            'y': random.randint(-180, 180 - obstacle_size),
            'color': random.choice(['red', 'green'])
        }
        obstacles.append(obstacle)

# Initialize turtle properties
drawer.speed(0)
drawer.up()

# Main game loop
running = True
generation = 1

bots = initialize_bots()
finish_line = {
    'x': random.randint(-180, 180 - finish_line_width),
    'y': random.randint(-180, 180 - finish_line_height)
}
obstacles = []
for _ in range(population_size):
    obstacle = {
        'x': random.randint(-180, 180 - obstacle_size),
        'y': random.randint(-180, 180 - obstacle_size),
        'color': random.choice(['red', 'green'])
    }
    obstacles.append(obstacle)

while running:
    processes = []
    for bot in bots:
        process = multiprocessing.Process(target=move_bot, args=(bot,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    draw_game_objects()
    time.sleep(0.001)  # Delay for smoother animation

    # Check if all bots reached the finish line
    if all(check_finish(bot) for bot in bots):
        update_brains()
        reset_game()
        generation += 1
        print(f"Generation {generation}")

        # Print the fitness scores
        fitness_scores = [bot['fitness'] for bot in bots]
        print(f"Fitness Scores: {fitness_scores}")

        time.sleep()
