from tensorflow.keras.preprocessing.image import ImageDataGenerator

def datagen(path = './data/', train_dir = 'train/', test_dir = 'test/', val_dir = 'validation/', aug = True):
    """
    Recieves the directory for tha data its divisions for train,
    test and validation.

    'aug' refers to if want to do data augmentation,
    by default its True

    returns:
    train_generator, validation_generator, test_generator
    """
    train_dir = path + train_dir
    test_dir = path + test_dir
    val_dir = path + val_dir

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.1
    )

    if aug != True:
        train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    validation_generator = train_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    return train_generator, validation_generator, test_generator