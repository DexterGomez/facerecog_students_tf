import matplotlib.pyplot as plt

def plot_history(history, plottittle="training_plot"):
    """
    Gets the results of 'history = model.fit()' to create
    a plot for accuracy and loss in each epoch.

    Saves the plots as .png files

    returns nothing
    """
    # Number of epochs
    epochs_range = range(1, len(history.history['accuracy']) + 1)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'])
    plt.plot(epochs_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(epochs_range)  # set x-ticks to be integers
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'])
    plt.plot(epochs_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs_range)  # set x-ticks to be integers
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f'{str(plottittle)}.png')

    print("DONE")