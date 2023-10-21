from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def evaluate_model(model, test_generator):
    """
    Recieves a model trained from VGG16 or MobileNet for face recognition
    and a test_generator to evaluate the model with accuracy and F-1 score.

    Returns accuarcy and f-1 as float
    """

    steps = test_generator.samples // test_generator.batch_size
    accuracy = []
    f1 = []

    for i in range(steps):
        X, y_true = next(test_generator)
        y_pred = model.predict(X)

        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        
        ac = accuracy_score(y_true, y_pred)
        f = f1_score(y_true, y_pred, average='weighted')  # 'weighted' if multi-class, otherwise 'binary'

        accuracy.append(ac)
        f1.append(f)

    accuracy = sum(accuracy)/steps
    f1 = sum(f1)/steps

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, f1