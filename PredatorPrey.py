import pygame
import numpy as np
import random
import math
import collections
import cProfile

# Graph settings
GRAPH_WIDTH = 300
GRAPH_HEIGHT = 100
FPS_HISTORY_LENGTH = 5000
POPULATION_HISTORY_LENGTH = 5000

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1500, 1000
NUM_PREDATORS = 40
NUM_PREY = 80
MAX_PREDATORS = 200
MAX_PREY = 300
AGENT_SIZE = 5


# Predators Constants
PREDATOR_CONSTANTS = {
    "energy_depletion_rate": 7.5,  # Base energy depletion rate per second
    "movement_energy_multiplier": 5.0,  # Energy depletion multiplier when moving
    "reproduction_time": 10,  # Reproduction time in seconds
    "movement_speed": 120,  # Movement speed multiplier
    "raycast_distance": 300,  # Distance for raycasting
    "fov": 1.5 * math.pi / 4,  # Field of view for predators (45 degrees)
    "color": (207, 48, 48)  # Color for predator visualization
}

# Prey Constants
PREY_CONSTANTS = {
    "energy_gain_rate": 1.0,  # Energy gain rate per second
    "movement_energy_multiplier": 5.0,  # Energy depletion multiplier when moving
    "reproduction_time": 5,  # Reproduction time in seconds
    "movement_speed": 100,  # Movement speed multiplier
    "raycast_distance": 150,  # Distance for raycasting
    "fov": 5 * math.pi / 3,  # Field of view for prey (120 degrees)
    "color": (60, 194, 60)  # Color for prey visualization
}


# Raycasting parameters
NUM_RAYCASTS = 18


# Colors
RAY_COLOR = (255, 255, 255)  # White color for ray visualization
POSITIVE_COLOR = (0, 200, 0)
NEGATIVE_COLOR = (200, 0, 0)

# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Initialize variables for graphing
show_graphs = False
fps_history = collections.deque(maxlen=FPS_HISTORY_LENGTH)
predator_population_history = collections.deque(maxlen=POPULATION_HISTORY_LENGTH)
prey_population_history = collections.deque(maxlen=POPULATION_HISTORY_LENGTH)

# Grid size for spatial partitioning
GRID_SIZE = 125  

# get position in the grid
def get_grid_position(x, y):
    return int(x // GRID_SIZE), int(y // GRID_SIZE)

non_empty_cells = set()
predator_cells = set()
prey_cells = set()

# add an agent to the grid
def add_to_grid(grid, agent):
    grid_pos = get_grid_position(agent.x, agent.y)
    if grid_pos not in grid:
        grid[grid_pos] = []
    grid[grid_pos].append(agent)
    non_empty_cells.add(grid_pos)  # Mark cell as non-empty
    if agent.is_predator:
        predator_cells.add(grid_pos)
    else:
        prey_cells.add(grid_pos)

# remove an agent from the grid
def remove_from_grid(grid, agent):
    grid_pos = get_grid_position(agent.x, agent.y)
    if grid_pos in grid:
        grid[grid_pos].remove(agent)
        if not grid[grid_pos]:  # If the cell is empty after removal
            del grid[grid_pos]
            non_empty_cells.discard(grid_pos)  # Mark cell as empty
        if agent.is_predator:
            predator_cells.discard(grid_pos)
        else:
            prey_cells.discard(grid_pos)

# retrun all cells that an agent can see
def get_cells_in_fov(x, y, angle, fov, max_distance, target_cells,is_predator):
    cells = set()
    half_fov = fov / 2

    max_distance_squared = max_distance * max_distance  # Use squared distances to avoid sqrt

    for cell_x, cell_y in target_cells:

        if get_grid_position(x,y) == (cell_x,cell_y):
            cells.add((cell_x, cell_y))
            continue


        # Calculate the center of the cell
        cell_center_x = cell_x * GRID_SIZE + GRID_SIZE / 2
        cell_center_y = cell_y * GRID_SIZE + GRID_SIZE / 2

        # Calculate vector from agent to cell center
        dx = cell_center_x - x
        dy = cell_center_y - y

        # Calculate the squared distance from agent to cell center
        distance_squared = dx * dx + dy * dy

        # Skip cells that are out of range
        if distance_squared > max_distance_squared *1.3:
            continue
        else:
            if not is_predator:
                cells.add((cell_x, cell_y)) # for prey just use distance as thier fov is close to 360
                continue

        # Calculate the angle between the agent's facing direction and the cell
        cell_angle = math.atan2(dy, dx)
        angle_diff = (cell_angle - angle + math.pi) % (2 * math.pi) - math.pi

        # Check if the cell center is within the FOV
        if abs(angle_diff) <= half_fov:
            cells.add((cell_x, cell_y))
            continue

        # If the center is not within the FOV, check if any corner of the bounding box is within the FOV
        cell_corners = [
            (cell_x * GRID_SIZE, cell_y * GRID_SIZE),  # Top-left corner
            (cell_x * GRID_SIZE + GRID_SIZE, cell_y * GRID_SIZE),  # Top-right corner
            (cell_x * GRID_SIZE, cell_y * GRID_SIZE + GRID_SIZE),  # Bottom-left corner
            (cell_x * GRID_SIZE + GRID_SIZE, cell_y * GRID_SIZE + GRID_SIZE)  # Bottom-right corner
        ]

        for corner_x, corner_y in cell_corners:
            dx = corner_x - x
            dy = corner_y - y
            distance_squared = dx * dx + dy * dy

            if distance_squared <= max_distance_squared:
                corner_angle = math.atan2(dy, dx)
                angle_diff = (corner_angle - angle + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_diff) <= half_fov:
                    cells.add((cell_x, cell_y))
                    break  # No need to check other corners

    return cells

# single raycast to check for agents 
def cast_ray(x, y, angle, max_distance, obstacles, target_type_predator):
    end_x = x + math.cos(angle) * max_distance
    end_y = y + math.sin(angle) * max_distance

    closest_distance_squared = max_distance * max_distance  # Use squared distances to avoid sqrt

    # loop through list of obstacles provided by the spatial partition
    for obstacle in obstacles:
        if obstacle.is_predator == target_type_predator:
            ox, oy = obstacle.x, obstacle.y
            dx = ox - x
            dy = oy - y
            distance_squared = dx * dx + dy * dy
            
            # if the obstacle is closer than the max raycast distace, perfrome a line intersect circle calculation
            if distance_squared <= closest_distance_squared:
                if line_intersects_circle(x, y, end_x, end_y, ox, oy, AGENT_SIZE):
                    closest_distance_squared = min(closest_distance_squared, distance_squared)

    closest_distance = math.sqrt(closest_distance_squared)
    normalized_distance = 1 - (closest_distance_squared / (max_distance * max_distance))
    return closest_distance, normalized_distance

# self explanatory
def line_intersects_circle(x1, y1, x2, y2, cx, cy, radius):
    # Calculate the vector from point 1 to point 2
    dx, dy = x2 - x1, y2 - y1

    # Vector from point 1 to the circle center
    fx, fy = x1 - cx, y1 - cy

    # Precompute terms
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius

    # Check if the line is outside the circle (early exit if no intersection)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False  # No intersection

    # Compute discriminant's square root
    discriminant_sqrt = math.sqrt(discriminant)

    # Calculate t values for intersection points
    t1 = (-b - discriminant_sqrt) / (2 * a)
    t2 = (-b + discriminant_sqrt) / (2 * a)

    # Check if any intersection points are within the segment
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

# draw energy and reproduction bars
def draw_status_bars(screen, agent, pos, width=300, height=20, padding=10):
    bar_x, bar_y = pos

    # Draw energy bar
    energy_ratio = agent.energy / 100.0  # Assuming max energy is 100
    energy_bar_width = energy_ratio * width
    pygame.draw.rect(screen, (255, 0, 0), (bar_x, bar_y, width, height))  # Red background
    pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, energy_bar_width, height))  # Green for actual energy

    # Draw reproduction timer bar
    bar_y += height + padding

    # Calculate the reproduction ratio correctly
    reproduction_ratio = agent.age / agent.constants["reproduction_time"]
    reproduction_bar_width = reproduction_ratio * width
    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, width, height))  # Grey background
    pygame.draw.rect(screen, (0, 0, 255), (bar_x, bar_y, reproduction_bar_width, height))  # Blue for timer

# class that represents nodes in the NN
class Node:
    def __init__(self, node_id, layer):
        self.node_id = node_id
        self.layer = layer
        self.value = 0.0
        self.bias = np.random.uniform(-0.1, 0.1)  # Initialize bias within a small range

# class that represents connections between nodes
class Connection:
    def __init__(self, in_node, out_node, weight=None, enabled=True):
        self.in_node = in_node  # Node ID
        self.out_node = out_node  # Node ID
        self.weight = weight if weight is not None else np.random.uniform(-1, 1)  # Initialize weight within a range
        self.enabled = enabled

    # mutates the weight of the connection 
    def mutate_weight(self, mutation_strength=0.1):
        if random.random() < 0.9:  # 90% chance to perturb weight
            self.weight += np.random.uniform(-1, 1) * mutation_strength
        else:  # 10% chance to assign new weight
            self.weight = np.random.uniform(-1, 1)

# class that represents the NEAT neural netwrok
class NeuralNetwork:
    def __init__(self):
        self.nodes = {}  # Node ID -> Node object
        self.connections = []  # List of Connection objects
        self.layer_nodes = {0: [], 1: [], 2: [], 3: [], 4: []}  # Nodes per layer
        self.input_nodes = [] 
        self.output_nodes = []
        self.next_node_id = 0
        self.visualization_mode = "activation" 
        # which mode of visualisation to be used.                                      
        # "activation" shows the activation values within the nodes and the weighted values between them
        # "weights_biases" shows the biases within the nodes and the weights between them

    def initialize_network(self, num_inputs, num_outputs):
        # Initialize input nodes in layer 0
        for _ in range(num_inputs):
            node_id = self.next_node_id
            new_node = Node(node_id, 0)
            self.nodes[node_id] = new_node
            self.input_nodes.append(node_id)
            self.layer_nodes[0].append(node_id)
            self.next_node_id += 1

        # Initialize output nodes in layer 4
        for _ in range(num_outputs):
            node_id = self.next_node_id
            new_node = Node(node_id, 4)
            self.nodes[node_id] = new_node
            self.output_nodes.append(node_id)
            self.layer_nodes[4].append(node_id)
            self.next_node_id += 1

        # Start with some random connections
        for _ in range(10):
            in_node = random.choice(self.input_nodes)
            out_node = random.choice(self.output_nodes)
            self.add_connection(in_node, out_node)
    # adds a connection between 2 nodes
    def add_connection(self, in_node, out_node, weight=None):
        if in_node == out_node:
            return  # Avoid self-connections

        in_layer = self.nodes[in_node].layer
        out_layer = self.nodes[out_node].layer

        if in_layer >= out_layer:
            return  # Ensure connections only go forward

        new_connection = Connection(in_node, out_node, weight)
        self.connections.append(new_connection)
    #removes a connection between 2 nodes
    def remove_connection(self):
        if not self.connections:
            return
        connection_to_remove = random.choice(self.connections)
        self.connections.remove(connection_to_remove)

    #adds a node in the middle of a connection
    def add_node(self):
        if not self.connections:
            return

        # chooses a random connection to split
        connection_to_split = random.choice([conn for conn in self.connections if conn.enabled]) 
        connection_to_split.enabled = False # disables the connection 

        # selects the apropriate layer that is between the two nodes
        new_layer = (self.nodes[connection_to_split.in_node].layer + self.nodes[connection_to_split.out_node].layer) // 2 
        new_node_id = self.next_node_id
        self.next_node_id += 1

        # adds a new node to the layer
        new_node = Node(new_node_id, new_layer)
        self.nodes[new_node_id] = new_node
        self.layer_nodes[new_layer].append(new_node_id) 

        # adds new connections between the new node and the nodes either side of the split connection 
        self.add_connection(connection_to_split.in_node, new_node_id, weight=1.0)
        self.add_connection(new_node_id, connection_to_split.out_node, weight=connection_to_split.weight)

    # removes a random node 
    def remove_node(self):
        if len(self.nodes) <= len(self.input_nodes) + len(self.output_nodes):
            return  # Do not remove input or output nodes

        node_id_to_remove = random.choice([node_id for node_id in self.nodes if node_id not in self.input_nodes + self.output_nodes])

        # Remove all connections involving this node
        self.connections = [conn for conn in self.connections if conn.in_node != node_id_to_remove and conn.out_node != node_id_to_remove]

        # Remove the node from its layer
        self.layer_nodes[self.nodes[node_id_to_remove].layer].remove(node_id_to_remove)

        # Remove the node itself
        del self.nodes[node_id_to_remove]

    # selects a random connection and calls its mutate weight function
    def mutate_weights(self):
        if not self.connections:
            return
        connection_to_mutate = random.choice(self.connections)
        connection_to_mutate.mutate_weight(mutation_strength=0.1)

    # mutates the bias of a random node
    def mutate_bias(self):
        if not self.nodes:
            return
        node_to_mutate = random.choice(list(self.nodes.values()))
        node_to_mutate.bias += np.random.uniform(-0.1, 0.1)

    # selects a random mutation 
    def mutate(self):
        # Define the probabilities for each mutation type
        mutation_types = {
            "add_node": 0.1,
            "remove_node": 0.05,
            "add_connection": 0.2,
            "remove_connection": 0.1,
            "mutate_weight": 0.3,
            "mutate_bias": 0.2,
            "no_mutation": 0.05,
        }

        # Normalize the probabilities
        total = sum(mutation_types.values())
        mutation_types = {key: val / total for key, val in mutation_types.items()}

        # Perform weighted random choice of mutation type
        mutation_choice = random.choices(list(mutation_types.keys()), weights=mutation_types.values())[0]

        if mutation_choice == "add_node":
            self.add_node()
        elif mutation_choice == "remove_node":
            self.remove_node()
        elif mutation_choice == "add_connection":
            self.add_connection(random.choice(list(self.nodes.keys())), random.choice(list(self.nodes.keys())))
        elif mutation_choice == "remove_connection":
            self.remove_connection()
        elif mutation_choice == "mutate_weight":
            self.mutate_weights()
        elif mutation_choice == "mutate_bias":
            self.mutate_bias()
        # If 'no_mutation' is chosen, do nothing.

    #activates the NN with inputs from the raycasts and returns the outputs to be used in movement
    def activate(self, inputs):
        # Set input values
        for i, input_value in enumerate(inputs):
            self.nodes[self.input_nodes[i]].value = input_value

        # Reset node values for all layers except the input layer
        for layer in range(1, 5):
            for node_id in self.layer_nodes[layer]:
                self.nodes[node_id].value = 0.0

        # Process each connection and calculate node outputs
        for conn in self.connections:
            if conn.enabled:
                self.nodes[conn.out_node].value += (self.nodes[conn.in_node].value * conn.weight)

        # Apply bias and tanh activation function to all nodes except input nodes
        for node in self.nodes.values():
            if node.layer > 0:  # Do not apply to input nodes
                node.value += node.bias
                node.value = np.tanh(node.value)  # Outputs between -1 and 1

        # Gather output values
        return [self.nodes[node_id].value for node_id in self.output_nodes]

    # renders a visualisation of the NN 
    def render(self, screen, inputs, pos, size=(300, 200), padding=10, background_color=(50, 50, 50)):
        self.activate(inputs)  # Get the activation values by running the network - netwrorks are not that complec so no harm in activating one of them twice 
        viz_x, viz_y = pos
        viz_width, viz_height = size
        viz_x += padding
        viz_y += padding
        viz_width -= 2 * padding
        viz_height -= 2 * padding

        # Draw the background
        pygame.draw.rect(screen, background_color, (viz_x - padding, viz_y - padding, viz_width + 2 * padding, viz_height + 2 * padding))

        node_radius = max(5, viz_width // 20)
        node_positions = {}

        # Position nodes in each layer
        for layer, nodes in self.layer_nodes.items():
            node_spacing_y = viz_height // (len(nodes) + 1)
            for i, node_id in enumerate(nodes):
                node_x = viz_x + layer * (viz_width // 4)
                node_y = viz_y + (i + 1) * node_spacing_y
                node_positions[node_id] = (node_x, node_y)

                node_value = self.nodes[node_id].value
                node_color = (0, 255, 0) if node_value >= 0 else (255, 0, 0)
                node_radius_scaled = max(3, int(node_radius * abs(node_value)))

                # Draw node based on the current visualization mode
                if self.visualization_mode == "activation":
                    pygame.draw.circle(screen, node_color, (int(node_x), int(node_y)), node_radius_scaled)
                else:  # "weights_biases" mode
                    bias_color = (0, 0, 255) if self.nodes[node_id].bias >= 0 else (255, 0, 0)
                    pygame.draw.circle(screen, bias_color, (int(node_x), int(node_y)), node_radius_scaled)
                    # Display bias value inside the node
                    font = pygame.font.SysFont(None, 18)
                    bias_text = font.render(f'{self.nodes[node_id].bias:.2f}', True, (255, 255, 255))
                    screen.blit(bias_text, (node_x - node_radius, node_y - node_radius))

        # Draw connections with the appropriate visualization mode
        for conn in self.connections:
            if conn.enabled:
                start_pos = node_positions.get(conn.in_node)
                end_pos = node_positions.get(conn.out_node)
                if start_pos and end_pos:
                    if self.visualization_mode == "activation":
                        activation_value = self.nodes[conn.in_node].value * conn.weight
                        color_intensity = int(min(max(255 * abs(activation_value), 0), 255))
                        color = (0, color_intensity, 0) if activation_value >= 0 else (color_intensity, 0, 0)
                        thickness = max(1, int(3 * abs(activation_value)))
                    else:  # "weights_biases" mode
                        weight_color = (0, 0, 255) if conn.weight >= 0 else (255, 0, 0)
                        thickness = max(1, int(3 * abs(conn.weight)))
                        color = weight_color

                    pygame.draw.line(screen, color, start_pos, end_pos, thickness)

    # self explanatory
    def toggle_visualization_mode(self):
        if self.visualization_mode == "activation":
            self.visualization_mode = "weights_biases"
        else:
            self.visualization_mode = "activation"


# class that represents the agent - should have inheritance but created more problems than it solved 
class Agent:
    def __init__(self, x, y, energy, is_predator=True, randomize_timer=True):
        self.x = x
        self.y = y
        self.energy = energy
        self.is_predator = is_predator
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.can_move = True

        # Select constants based on whether the agent is a predator or prey
        constants = PREDATOR_CONSTANTS if is_predator else PREY_CONSTANTS
        self.constants = constants  # Store the selected constants

        self.reproduction_time = constants["reproduction_time"]
        self.net = NeuralNetwork()
        self.net.initialize_network(NUM_RAYCASTS, 2)  # Initialize with raycasts as inputs and 2 outputs (rotation, speed)
        self.time_since_last_raycast = random.uniform(0, 0.2)  # Stagger raycasts timing initially
        self.cached_inputs = [0] * NUM_RAYCASTS  # cached inputs to use for the NN during frames where no new raycasts are used
        self.dead = False
        

        self.age = random.uniform(0, self.reproduction_time) if randomize_timer else 0 # staggers initial age to stop all agents reproducing at the same time

    # handles movement logic
    def move(self, delta_time, grid, target_cells, selected):
        self.time_since_last_raycast += delta_time

        #perfroms raycasts if enough time has passed since the last
        if self.time_since_last_raycast >= 0.2:  # 5 times per second
            inputs = self.raycast(grid, target_cells)
            self.cached_inputs = inputs
            self.time_since_last_raycast = 0  # Reset timer


        inputs = self.cached_inputs

        if selected:
            print(f"Agent: {self}, Inputs: {inputs}")

        activations = self.net.activate(inputs) # activates the neural netwrok to get the output values

        if selected:
            print(f"Agent: {self}, Activations: {activations}")

        outputs = activations

        # moves and rotates based on inputs - could have added momentum here but the agent would not be doing exaclty what the NN wanted
        rotation_output = outputs[0]
        self.speed = max(-1, outputs[1]) * self.constants["movement_speed"]
        rotation = rotation_output * math.pi if rotation_output != 0 else 0

        self.angle += rotation * delta_time

        self.x += math.cos(self.angle) * self.speed * delta_time
        self.y += math.sin(self.angle) * self.speed * delta_time

        # keeps the agents on the screen by wrapping the world space
        self.x %= SCREEN_WIDTH
        self.y %= SCREEN_HEIGHT

        # reduces movement speed if its low on energy 
        if not self.can_move:
            self.speed *= 0.2

        
        
    # performes multiple raycats to gather the inputs for the NN
    def raycast(self, grid, target_cells):
        raycast_distance = self.constants["raycast_distance"]
        fov = self.constants["fov"]
        half_fov = fov / 2
        start_angle = self.angle - half_fov
        step = fov / (NUM_RAYCASTS - 1)

        #retrieves the relevant cells for the raycasts 
        cells_in_fov = get_cells_in_fov(self.x, self.y, self.angle, fov, raycast_distance, target_cells,self.is_predator)
        nearby_agents = []
        for cell in cells_in_fov:
            if cell in grid:
                nearby_agents.extend(grid[cell])

        #casts a ray for based on the number of raycasts and fov
        distances = []
        for i in range(NUM_RAYCASTS):
            angle = start_angle + i * step
            _, normalized_distance = cast_ray(self.x, self.y, angle, raycast_distance, nearby_agents, target_type_predator=not self.is_predator)
            distances.append(normalized_distance)

        return distances

    # logic to handle reproduction 
    def reproduce(self, agent_count):
        if (self.is_predator and agent_count >= MAX_PREDATORS) or (not self.is_predator and agent_count >= MAX_PREY):
            return None  # Do not reproduce if max agents reached

        new_agent = Agent(self.x, self.y, 75, is_predator=self.is_predator, randomize_timer=False)
        
        # Clone the neural network
        new_agent.net = NeuralNetwork()
        new_agent.net.nodes = {node_id: Node(node_id, node.layer) for node_id, node in self.net.nodes.items()}
        new_agent.net.connections = [Connection(conn.in_node, conn.out_node, conn.weight, conn.enabled) for conn in self.net.connections]
        new_agent.net.input_nodes = self.net.input_nodes[:]
        new_agent.net.output_nodes = self.net.output_nodes[:]
        new_agent.net.layer_nodes = {layer: node_ids[:] for layer, node_ids in self.net.layer_nodes.items()}
        new_agent.net.next_node_id = self.net.next_node_id

        # Mutate the new agent's network
        new_agent.net.mutate()
        
        self.age = random.random()  # Reset the age after reproduction
        self.reproduction_time = self.constants["reproduction_time"]

        print(f"Reproduced new agent at ({new_agent.x}, {new_agent.y})")

        return new_agent

    # update loop of the agentv
    def update(self, delta_time, grid, target_cells, selected, agent_count):
        if self.dead:
            return False # kills the agent
        
        #moves the agent which also triggers the NN to activate
        self.move(delta_time, grid, target_cells, selected)
        self.age += delta_time
        self.age = min(self.age, self.constants["reproduction_time"])

        # Retrieve the grid position
        grid_pos = get_grid_position(self.x, self.y)

        # Get nearby agents 
        neighboring_cells = [
            grid_pos,
            (grid_pos[0] - 1, grid_pos[1]), (grid_pos[0] + 1, grid_pos[1]),  # Horizontal neighbors
            (grid_pos[0], grid_pos[1] - 1), (grid_pos[0], grid_pos[1] + 1),  # Vertical neighbors
            (grid_pos[0] - 1, grid_pos[1] - 1), (grid_pos[0] + 1, grid_pos[1] + 1),  # Diagonal neighbors
            (grid_pos[0] - 1, grid_pos[1] + 1), (grid_pos[0] + 1, grid_pos[1] - 1)
        ]
        
        nearby_agents = []
        for cell in neighboring_cells:
            if cell in grid:
                nearby_agents.extend(grid[cell])

        for agent in nearby_agents:
            if agent == self:
                continue

            # Calculate distance between agents
            dx = agent.x - self.x
            dy = agent.y - self.y
            distance_squared = dx * dx + dy * dy
            min_distance = AGENT_SIZE * 2

            if distance_squared < min_distance * min_distance:
                distance = math.sqrt(distance_squared)
                overlap = min_distance - distance

                # Handle collision with the same type (push away)
                if agent.is_predator == self.is_predator:
                    # Calculate push direction
                    if distance > 0:
                        push_x = (dx / distance) * (overlap / 2)
                        push_y = (dy / distance) * (overlap / 2)
                    else:
                        # Handle exact overlap case (random direction push) - happens during reproduction
                        push_x = random.uniform(-1, 1) * (overlap / 2)
                        push_y = random.uniform(-1, 1) * (overlap / 2)
                    
                    # Apply push
                    self.x -= push_x
                    self.y -= push_y
                    agent.x += push_x
                    agent.y += push_y
                
                # Handle predator-prey interaction (predator eats prey)
                elif self.is_predator and not agent.is_predator:
                    if distance_squared < AGENT_SIZE * AGENT_SIZE * 1.5:
                        self.energy += 40
                        self.age += 1
                        self.energy = min(self.energy, 100)
                        agent.dead = True  # Mark prey as dead

        true_speed = abs(self.speed)
        
        #handles energy depletion and regeneration          
        if self.is_predator: # predators lose energy over time, with them losing more when they move
            self.energy -= (self.constants["energy_depletion_rate"] + self.constants["movement_energy_multiplier"] * true_speed / 100) * delta_time
        else:                # prey gain energy over time, with them also losing energy while moving
            self.energy -= (-self.constants["energy_gain_rate"] + self.constants["movement_energy_multiplier"] * true_speed / 100) * delta_time
            self.energy = min(self.energy, 100)


        if self.energy <= 1:
            if self.is_predator:
                return False  # Predator dies if it runs out of energy 
            else:
                self.can_move = False # prey have their movement speed reduced 
        elif self.energy >= 10:
            self.can_move = True # prey can move at full speed once they regain energy 

        if self.age >= self.constants["reproduction_time"] and ((self.is_predator and agent_count < MAX_PREDATORS) or (not self.is_predator and agent_count < MAX_PREY)):
            return self.reproduce(agent_count)  # Return the new Agent instance if reproduction occurs

        return True  # Agent survives

                    
    # draws the agent
    def draw(self, screen, show_rays=False, grid=None, target_cells=None):
        color = self.constants["color"]
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), AGENT_SIZE) # agent body

        end_x = self.x + math.cos(self.angle) * (AGENT_SIZE + 10)
        end_y = self.y + math.sin(self.angle) * (AGENT_SIZE + 10)
        pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 2) # line representing agent direction 

        if show_rays and grid is not None:
            self.draw_rays(screen, grid, target_cells) 

    # draws raycasts if the agent is selected
    def draw_rays(self, screen, grid, target_cells):
        raycast_distance = self.constants["raycast_distance"]
        fov = self.constants["fov"]
        half_fov = fov / 2
        start_angle = self.angle - half_fov
        step = fov / (NUM_RAYCASTS - 1)
        target_type_predator = not self.is_predator

        # Get all cells within the FOV
        cells_in_fov = get_cells_in_fov(self.x, self.y, self.angle, fov, raycast_distance, target_cells,self.is_predator)
        nearby_agents = []

        # Highlight cells within the FOV
        for cell in cells_in_fov:
            cell_x, cell_y = cell
            pygame.draw.rect(screen, (0, 0, 255, 50), (cell_x * GRID_SIZE, cell_y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
            if cell in grid:
                for agent in grid[cell]:
                    if agent.is_predator == target_type_predator:
                        nearby_agents.append(agent)
                        # Highlight the agents being considered by the raycasts
                        pygame.draw.circle(screen, (255, 255, 0), (int(agent.x), int(agent.y)), AGENT_SIZE + 2, 2)

        # Perform raycasting and draw rays
        for i in range(NUM_RAYCASTS):
            angle = start_angle + i * step
            distance, _ = cast_ray(self.x, self.y, angle, raycast_distance, nearby_agents, target_type_predator=target_type_predator)
            end_x = self.x + math.cos(angle) * distance
            end_y = self.y + math.sin(angle) * distance
            pygame.draw.line(screen, RAY_COLOR, (self.x, self.y), (end_x, end_y))
            print(f"Ray {i} distance: {distance}")

#spawns a number of agents in random locations
def create_agents(num_agents, is_predator):
    agents = []
    for _ in range(num_agents):
        x = random.uniform(0, SCREEN_WIDTH)
        y = random.uniform(0, SCREEN_HEIGHT)
        energy = 100.0
        agent = Agent(x, y, energy, is_predator=is_predator)
        agents.append(agent)
    return agents

# gets an agent at the mouse position 
def get_agent_at_position(agents, pos):
    for agent in agents:
        if math.hypot(agent.x - pos[0], agent.y - pos[1]) <= AGENT_SIZE:
            return agent
    return None

# draws a graph showing FPS history 
def draw_fps_graph(screen, fps_history, graph_x=10, graph_y=10, graph_width=GRAPH_WIDTH, graph_height=GRAPH_HEIGHT):
    # Draw static background and border only once
    static_background = pygame.Surface((graph_width, graph_height))
    static_background.fill((50, 50, 50))
    pygame.draw.rect(static_background, (255, 255, 255), (0, 0, graph_width, graph_height), 2)

    # Calculate scaling factors
    max_fps = max(fps_history) if fps_history else 60
    effective_max_fps = max_fps / 0.75
    scale_x = graph_width / len(fps_history)
    scale_y = graph_height / effective_max_fps

    # Create a surface to draw the graph on
    graph_surface = static_background.copy()
    
    # Draw the graph using a batch line drawing
    points = [(i * scale_x, graph_height - fps_history[i] * scale_y) for i in range(len(fps_history))]
    pygame.draw.lines(graph_surface, (0, 255, 0), False, points, 2)

    # Render the current FPS label
    font = pygame.font.SysFont(None, 18)
    current_fps = fps_history[-1] if fps_history else 0
    fps_label_text = font.render(f'FPS: {int(current_fps)}', True, (255, 255, 255))
    graph_surface.blit(fps_label_text, (5, 5))

    # Blit the graph surface onto the main screen
    screen.blit(graph_surface, (graph_x, graph_y))


# draws graphs showing population history for both predators and prey
def draw_population_graph(screen, history, color, graph_x, graph_y, max_population, label, graph_width=GRAPH_WIDTH, graph_height=GRAPH_HEIGHT):
    # Draw static background and border only once
    static_background = pygame.Surface((graph_width, graph_height))
    static_background.fill((50, 50, 50))
    pygame.draw.rect(static_background, (255, 255, 255), (0, 0, graph_width, graph_height), 2)
    
    # Calculate scaling factors
    effective_max_population = max_population / 0.75
    scale_x = graph_width / len(history)
    scale_y = graph_height / effective_max_population

    # Create a surface to draw the graph on
    graph_surface = static_background.copy()
    
    # Draw the graph using a batch line drawing
    points = [(i * scale_x, graph_height - history[i] * scale_y) for i in range(len(history))]
    pygame.draw.lines(graph_surface, color, False, points, 2)
    
    # Render the current population label
    font = pygame.font.SysFont(None, 18)
    population_label_text = font.render(f'{label}: {int(history[-1])}', True, (255, 255, 255))
    graph_surface.blit(population_label_text, (5, 5))

    # Blit the graph surface onto the main screen
    screen.blit(graph_surface, (graph_x, graph_y))


def main():
    # lists containing all agents 
    predators = create_agents(NUM_PREDATORS, True)
    prey = create_agents(NUM_PREY, False)

    global show_graphs

    selected_agent = None
    non_empty_cells.clear()
    predator_cells.clear()
    prey_cells.clear()

    # main loop 
    running = True
    while running:
        delta_time = clock.tick(144) / 1000.0  # Convert milliseconds to seconds
        screen.fill((95, 97, 95))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # handle user input
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                selected_agent = get_agent_at_position(predators + prey, pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:  # Toggle graph visibility with 'G' key
                    show_graphs = not show_graphs
                elif event.key == pygame.K_n:  # Toggle visualization mode with 'N' key
                    if selected_agent:
                        selected_agent.net.toggle_visualization_mode()

        fps_history.append(clock.get_fps())
        predator_population_history.append(len(predators))
        prey_population_history.append(len(prey))

        grid = {}
        non_empty_cells.clear()  # Clear the set of non-empty cells before each frame
        predator_cells.clear()
        prey_cells.clear()

        for agent in predators + prey:
            add_to_grid(grid, agent)

        new_predators = []
        new_prey = []

        # update predators
        for predator in predators[:]:
            is_selected = predator == selected_agent
            result = predator.update(delta_time, grid, prey_cells, is_selected, len(predators))

            
            if not result:
                predators.remove(predator)
                if selected_agent == predator:
                    selected_agent = None  
            # adds the new agent to a list and changes the selcted agent to it if its parent was selected (to be able to track evolution of NN)
            elif isinstance(result, Agent):
                if len(predators) + len(new_predators) < MAX_PREDATORS:
                    new_predators.append(result)
                    if predator == selected_agent:
                        selected_agent = result
        # update prey
        for single_prey in prey[:]:
            is_selected = single_prey == selected_agent
            result = single_prey.update(delta_time, grid, predator_cells, is_selected, len(prey))

            # Clear the selected agent if it dies
            if not result:
                prey.remove(single_prey)
                if is_selected:
                    selected_agent = None  
            # adds the new agent to a list
            elif isinstance(result, Agent):
                if len(prey) + len(new_prey) < MAX_PREY:
                    new_prey.append(result)
                    if prey == selected_agent:
                        selected_agent = result

        # adds new agents to the correct lists
        predators.extend(new_predators)
        prey.extend(new_prey)

        # draws all agents
        for agent in predators + prey:
            is_selected = agent == selected_agent
            agent.draw(screen, show_rays=is_selected, grid=grid, target_cells=prey_cells if agent.is_predator else predator_cells)

        # draws a visualisation of the NN of the selcted agent, along with bars of their energy and reproduction timers
        if selected_agent:
            inputs = selected_agent.raycast(grid, prey_cells if selected_agent.is_predator else predator_cells)
            # Draw the status bars
            status_bar_position = (10, SCREEN_HEIGHT - SCREEN_HEIGHT // 4 - 50)
            draw_status_bars(screen, selected_agent, status_bar_position)
            
            # Draw the neural network below the status bars
            nn_position = (10, SCREEN_HEIGHT - SCREEN_HEIGHT // 4 - 10)
            selected_agent.net.render(screen, inputs, nn_position)

        # draws the graphs
        if show_graphs:
            draw_fps_graph(screen, fps_history)
            max_predators = max(predator_population_history) if predator_population_history else MAX_PREDATORS
            max_prey = max(prey_population_history) if prey_population_history else MAX_PREY
            draw_population_graph(screen, predator_population_history, (255, 0, 0), 10, 120, max_predators, "Predators")
            draw_population_graph(screen, prey_population_history, (0, 255, 0), 10, 240, max_prey, "Prey")
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    profiler.dump_stats("profile_output.prof")  # Save profile data

    #snakeviz profile_output.prof
