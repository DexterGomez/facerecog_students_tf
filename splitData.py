import os
import shutil
from random import shuffle

def split_dataset(data_dir):
    """
    Given a directory with data in divided in a directory
    for each class it splits it into training, val, and test.

    THERE MUST BE 300 IMAGES IN EACH CLASS TO WORK.
    THE DIVISION WILL BE:
    250 for training
    25 for validation
    25 for test
    """
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist!")
        return
    
    # Get all person directories in the data directory
    persons = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Create train, validation, and test directories if they don't exist
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    for directory in [train_dir, validation_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Process each person's directory
    for person in persons:
        # Skip if it's train, validation, or test directory
        if person in ['train', 'validation', 'test']:
            continue
        
        person_dir = os.path.join(data_dir, person)
        person_images = [f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))]
        
        # Shuffle the images to ensure randomness
        shuffle(person_images)
        
        # Split images
        train_images = person_images[:250]
        validation_images = person_images[250:275]
        test_images = person_images[275:300]
        
        # Function to move files
        def move_files(files, target_dir):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for f in files:
                shutil.move(os.path.join(person_dir, f), os.path.join(target_dir, f))
        
        # Move images to respective directories
        move_files(train_images, os.path.join(train_dir, person))
        move_files(validation_images, os.path.join(validation_dir, person))
        move_files(test_images, os.path.join(test_dir, person))
        
        # Remove original person directory if empty
        if not os.listdir(person_dir):
            os.rmdir(person_dir)

data_directory = "./data"
split_dataset(data_directory)
