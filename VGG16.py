import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

from datagen import datagen
from plothistory import plot_history
from evaluate import evaluate_model

N_PERSONS = 16
EPOCHS = 8
DATA_AUGMENTATION = True

MODEL_NAME = 'model_Vgg16_ep8_augT'

# Data preparation
train_generator, validation_generator, test_generator = datagen(aug=DATA_AUGMENTATION)

# Load model architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(N_PERSONS, activation='softmax')  # 15 classes for 15 friends
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(f'{MODEL_NAME}.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks
)

plot_history(history=history, plottittle=f"training_{MODEL_NAME}_plot")
evaluate_model(model=model, test_generator=test_generator)