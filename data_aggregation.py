# data aggregation
def data_aggregation():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    #Training Set
    train_set = train_datagen.flow_from_directory('train',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')
    #Validation Set
    test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64,64),
                                            batch_size = 32,
                                            class_mode='binary',
                                            shuffle=False)
    return train_set, test_set