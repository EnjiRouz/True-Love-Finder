"""
Данный скрипт производит создание модели нейронной сети на основе собранного с помощью image_generator.py датасета
(используется как train, так и test выборка)

Данный скрипт можно использовать для генерации моделей нейросетей для классификации для любых классов
в ваших конкретных случаях, заменив папки 'like' и 'dislike' на другие, а также изменив в соответствии с
названиями классов переменную 'variations' в данном скрипте

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# импорт библиотеки для вывода лога
from contextlib import redirect_stdout

# конфигурация
img_width, img_height = 200, 200
input_depth = 3
train_data_directory = 'generated-images-for-model/train'
testing_data_directory = 'generated-images-for-model/test'
epochs = 100
batch_size = 32

train_data = ImageDataGenerator(rescale=1 / 255)
test_data = ImageDataGenerator(rescale=1 / 255)

train_generator = train_data.flow_from_directory(
    train_data_directory,
    color_mode='rgb',  # использовать grayscale, если нужно использовать чёрно-белые изображения
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

testing_generator = test_data.flow_from_directory(
    testing_data_directory,
    color_mode='rgb',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# определение порядка данных входных изображений
if keras.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

model = Sequential()

# создание 2 каскадов свёрточных слоёв для выделения характерных признаков изображения
# создание первого свёрточного слоя
# используется ReLU, поскольку является менее ресурсоёмкой и обеспечивает высокую скорость обучения
model.add(Convolution2D(32, (3, 3), input_shape=input_shape_val, activation="relu", padding="same"))

# создание второго свёрточного слоя
model.add(Convolution2D(32, (3, 3), activation="relu"))

# создание слоя подвыборки (выбор максимального значения)
model.add(MaxPooling2D(pool_size=(2, 2)))

# создание слоя регуляризации (предотвращение/снижение переобучения посредством
# случайным образом выключенных нейронов, вероятность передаётся в качестве параметра)
model.add(Dropout(0.25))

# создание третьего свёрточного слоя с увеличенным числом карт признаков
model.add(Convolution2D(64, (3, 3), activation="relu", padding="same"))

# создание четвёртого свёрточного слоя с увеличенным числом карт признаков
model.add(Convolution2D(64, (3, 3), activation="relu"))

# создание второго слоя подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# создание второго слоя регуляризации
model.add(Dropout(0.25))

# создание классификатора, который по найденным признакам отнесёт изображение к определённому классу
# преобразование из двумерного вида в плоский
model.add(Flatten())

# создание полносвязного слоя
model.add(Dense(512, activation="relu"))

# создание слоя регуляризации
model.add(Dropout(0.5))

# создание выходного слоя с использованием выходных нейронов по количеству классов (train_generator.num_classes)
model.add(Dense(train_generator.num_classes, activation='softmax', name='output_tensor'))

# компиляция сети (используется categorical, т.к. существует несколько классов) с Stochastic Gradient Descent
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# обучение сети и определение валидационной выборки
model.fit_generator(
    train_generator,
    # число итераций за эпоху
    steps_per_epoch=np.floor(train_generator.n / batch_size),
    epochs=epochs,
    validation_data=testing_generator,
    validation_steps=np.floor(testing_generator.n / batch_size))

print("Training is done!")

# проверка точности модели на тренировочных данных
train_scores = model.evaluate_generator(train_generator, train_generator.n)

# проверка точности модели на тестовых данных
test_scores = model.evaluate_generator(testing_generator, testing_generator.n)

# сохранение модели нейросети
model.save('true-love-model.h5')
print("Model is successfully stored and tested!")

# запись данных о структуре нейросети в файл (можно также указывать любые другие данные, я это использую как лог)
with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        print("Accuracy on training data: %.2f%%" % (train_scores[1] * 100))
        print("Accuracy of work on test data: %.2f%%" % (test_scores[1] * 100))
        model.summary()
