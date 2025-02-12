import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def generate_candidate_pattern(shape):
        # Define the size of the pattern and the limited region
    pattern_size = (8, 8)  # Adjust the pattern size according to your needs
    limited_region = (20, 20, 28, 28)  # Define the region (top-left x, top-left y, width, height)

        # Generate a random pattern limited to the specified region
    pattern = np.zeros((28, 28))  # Create an empty pattern
    limited_pattern = np.random.randint(0, 256, pattern_size)  # Generate the limited pattern
    pattern[limited_region[1]:limited_region[1]+limited_region[3], limited_region[0]:limited_region[0]+limited_region[2]] = limited_pattern


    return pattern

def generate_optimal_trigger_pattern(x_non_target, x_target, model, layer_idx, iterations):
    # Define the dissimilarity measure
    def dissimilarity_measure(x1, x2):
        if x1 is None or x2 is None:
            return 0.0
        if x1.shape[0] != x2.shape[0]:
            # Resize the tensors to have the same number of samples
            min_samples = min(x1.shape[0], x2.shape[0])
            x1 = x1[:min_samples]
            x2 = x2[:min_samples]
            
            # Calculate the mean squared error between x1 and x2
        return tf.reduce_mean(tf.reduce_sum(tf.square(x1 - x2), axis=1))
    
    # Get the intermediate representations of non-target samples
    print(len(model.layers))
    features_non_target = model.layers[layer_idx].output
    intermediate_model_non_target = keras.Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    features_non_target = intermediate_model_non_target.predict(x_non_target)
    
    features_target = model.layers[layer_idx].output
    intermediate_model_target = keras.Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    features_target = intermediate_model_target.predict(x_target)
    
    # Perform optimization to find the optimal trigger pattern
    optimal_trigger_pattern = None
    optimal_loss = float('inf')
    
    for i in range(iterations):
        print(i)
        # Generate a candidate trigger pattern
        shape = (28, 28)  
        candidate_trigger_pattern = generate_candidate_pattern(shape)
        
        # Compute the dissimilarity between non-target and target samples
        loss = dissimilarity_measure(features_non_target, features_target)
        
        # Update the optimal trigger pattern if the loss is minimized
        if loss < optimal_loss:
            optimal_trigger_pattern = candidate_trigger_pattern
            optimal_loss = loss
    #print(optimal_loss)
    return optimal_trigger_pattern

def generate_optimized_pattern(layer_idx, target_class, iterations = 100):  #Saves an optimized trigger pattern
    x_train = np.load('datasets/teacher_x_train.npy')
    teacher_model = keras.models.load_model('trained models/teacher_model.h5')

    student_x_train = np.load('datasets/student_x_train.npy')
    student_y_train = np.load('datasets/student_y_train.npy')
    # Load the student_x_train dataset
    target_indices = np.where(student_y_train == target_class)[0]
    x_target = student_x_train[target_indices]


    # Preprocess the datasets
    x_non_target = x_train / 255.0
    x_target = x_target / 255.0

    x_target = x_target[:len(x_non_target)]
    
    # Generate the optimal trigger pattern for the target class
    print(teacher_model.layers[layer_idx].output_shape)
    optimal_trigger = generate_optimal_trigger_pattern(x_non_target, x_target, teacher_model, layer_idx, iterations)
    np.save('optimized_pattern.npy', optimal_trigger)

    plt.imshow(np.load('optimized_pattern.npy'), cmap='gray')
    plt.title('Backdoor Attack Optimal Pattern')
    plt.axis('off')
    plt.show()
    
def generate_random_pattern():
    pattern = np.random.randint(0, 256, (28, 28))
    print(pattern)
    np.save('random_pattern.npy', pattern)

    plt.imshow(pattern, cmap='gray')
    plt.title('Backdoor Attack Pattern')
    plt.axis('off')
    plt.show()

def generate_limited_pattern(): #Saves a corner-limited random pattern
    # Define the size of the pattern and the limited region
    pattern_size = (8, 8)  # Adjust the pattern size according to your needs
    limited_region = (20, 20, 28, 28)  # Define the region (top-left x, top-left y, width, height)

    # Generate a random pattern limited to the specified region
    pattern = np.zeros((28, 28))  # Create an empty pattern
    limited_pattern = np.random.randint(0, 256, pattern_size)  # Generate the limited pattern
    pattern[limited_region[1]:limited_region[1]+limited_region[3], limited_region[0]:limited_region[0]+limited_region[2]] = limited_pattern


    # Save the pattern
    np.save('limited_pattern.npy', pattern)

    plt.imshow(np.load('limited_pattern.npy'), cmap='gray')
    plt.title('Backdoor Attack Limited Pattern')
    plt.axis('off')
    plt.show()
    
def calculate_average_pattern(patterns):
    average_pattern = np.zeros((28, 28), dtype=np.float32)

    for i in range(28):
        for j in range(28):
            pixel_sum = 0
            for image in patterns:
                print(type(image))
                pixel_sum += image[i, j]
            average_pixel_value = pixel_sum / len(patterns)
            average_pattern[i, j] = average_pixel_value

    return average_pattern

def generate_average_ten():
    patterns = []
    for x in range(5,10):
        patterns.append(generate_optimized_pattern(5,x,10000))
    average_ten = calculate_average_pattern(patterns)
    #Save the pattern
    np.save('average_ten.npy', average_ten)

    plt.imshow(np.load('average_ten.npy'), cmap='gray')
    plt.title('average_ten Pattern')
    plt.axis('off')
    plt.show()

def generate_blank_pattern():
    # Define the size of the pattern and the limited region
    pattern_size = (8, 8)  # Adjust the pattern size according to your needs
    limited_region = (20, 20, 28, 28)  # Define the region (top-left x, top-left y, width, height)

    # Generate a random pattern limited to the specified region
    pattern = np.zeros((28, 28))  # Create an empty pattern
    limited_pattern = np.zeros(pattern_size)
    pattern[limited_region[1]:limited_region[1]+limited_region[3], limited_region[0]:limited_region[0]+limited_region[2]] = limited_pattern


    # Save the pattern
    np.save('blank_pattern.npy', pattern)

    plt.imshow(np.load('blank_pattern.npy'), cmap='gray')
    plt.title('Blank Pattern')
    plt.axis('off')
    plt.show()

def display_image(i_path):
    plt.imshow(np.load(i_path), cmap='gray')
    plt.title(i_path)
    plt.axis('off')
    plt.show()

generate_limited_pattern()
#generate_blank_pattern()
# generate_optimized_pattern(3, 8, 1000000)
#generate_average_ten()

display_image('limited_pattern.npy')
# display_image('optimized_pattern.npy')