import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from datagen import datagen
from plothistory import plot_history
from evaluate import evaluate_model

N_PERSONS = 16
EPOCHS = 4
DATA_AUGMENTATION = False

MODEL_NAME = 'model_MbNV2_ep4_augF'

# Data preparation
train_generator, validation_generator, test_generator = datagen(aug=DATA_AUGMENTATION)

# Load model architecture
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(N_PERSONS, activation="softmax")
])



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(f'{MODEL_NAME}.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs = EPOCHS,
    steps_per_epoch = train_generator.samples // train_generator.batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks
)

plot_history(history=history, plottittle=f"training_{MODEL_NAME}_plot")
evaluate_model(model=model, test_generator=test_generator)