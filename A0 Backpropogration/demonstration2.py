import random
import string
import time
import os

# Function to clear the console for better visualization
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Function to simulate the evolution of characters into target letters
def evolve_to_letters(target, steps=15, sleep_time=0.05):
    # Start with a random string of characters
    current = ''.join(random.choice(string.ascii_letters + string.punctuation + ' ') for _ in target)
    
    # Display initial random string
    print("Starting String:")
    print(current)
    time.sleep(1)
    
    for step in range(steps):
        clear_console()
        print("Step:", step + 1)
        next_state = []
        
        for i, target_char in enumerate(target):
            # Gradually "evolve" towards the target letter
            if current[i] != target_char:
                if random.random() > step / steps:  # Evolve the character step by step
                    next_state.append(random.choice(string.ascii_letters + string.punctuation + ' '))
                else:
                    next_state.append(target_char)
            else:
                next_state.append(target_char)
        
        current = ''.join(next_state)
        print(current)
        time.sleep(sleep_time)

# Example usage
target_string = "BACKPROP"
evolve_to_letters(target_string)
