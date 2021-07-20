from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
import PIL

train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory('data/training/', shuffle=True, batch_size=32, target_size=(24, 24),
                                          class_mode='binary')
valid_dataset = train.flow_from_directory('data/validation/', shuffle=True, batch_size=32, target_size=(24, 24),
                                          class_mode='binary')

SPE = len(train_dataset.classes) // 32
VS = len(valid_dataset.classes) // 32
print(SPE, VS)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),

    # 32 convolution filters used each of size 3x3
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),

    # 64 convolution filters used each of size 3x3
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_dataset, validation_data=valid_dataset, epochs=10, steps_per_epoch=SPE)

model.save('cnnCat2.h5')