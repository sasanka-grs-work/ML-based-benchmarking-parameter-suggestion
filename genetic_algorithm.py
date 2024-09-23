import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Enable multi-GPU strategy for distributed training
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])

# Define your data
input_shape = (5,)  # Example: 5 input parameters
num_configurations = 1000  # Example: 1000 data points

# Random example data
X_train = np.random.rand(num_configurations, *input_shape)
y_train = np.random.rand(num_configurations)  # Performance score

# Normalize the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Genetic Algorithm Parameters
population_size = 100
num_generations = 50
mutation_rate = 0.1
crossover_rate = 0.7

# Create a simple feedforward neural network model
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='linear')  # Predicting the score
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize the model within the strategy scope
with strategy.scope():
    model = create_model(input_shape)

# Train the model
with strategy.scope():
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=64)

# Genetic Algorithm Functions
def initialize_population():
    # Initialize a population with random parameters
    return np.random.rand(population_size, *input_shape)

def evaluate_population(population):
    # Predict performance (S) for each parameter set in the population
    population_scaled = scaler.transform(population)
    predictions = model.predict(population_scaled)
    return predictions

def selection(population, fitness):
    # Select the top half of the population based on fitness
    indices = np.argsort(fitness)[-population_size//2:]
    return population[indices]

def crossover(parent1, parent2):
    # Perform crossover between two parents
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, input_shape[0])
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    else:
        return parent1, parent2

def mutation(individual):
    # Mutate an individual
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(input_shape[0])
        individual[mutation_point] = np.random.rand()
    return individual

# Main Genetic Algorithm Loop
def genetic_algorithm():
    # Initialize population
    population = initialize_population()

    for generation in range(num_generations):
        # Evaluate the population
        fitness = evaluate_population(population)
        
        # Select the best individuals
        selected_population = selection(population, fitness)

        # Generate new population via crossover and mutation
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))

        population = np.array(new_population)

        # Logging progress
        print(f"Generation {generation+1}/{num_generations}, Best Fitness: {np.max(fitness)}")

    # Return the best individual
    best_fitness = evaluate_population(population)
    best_individual = population[np.argmax(best_fitness)]
    
    return best_individual

# Running the Genetic Algorithm
with strategy.scope():
    best_parameters = genetic_algorithm()

# Output the best parameters found
print("Best Parameters: ", best_parameters)
