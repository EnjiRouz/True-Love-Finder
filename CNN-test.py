"""
Данный скрипт производит проверку предсказаний нейронной сети на основе исходных файлов
(либо лежащих в конкретной директории, разделённой на папки классов)

Данный скрипт можно использовать для проверки моделей нейросетей для классификации для любых классов
в ваших конкретных случаях, заменив папки 'like' и 'dislike' на другие, а также изменив в соответствии с
названиями классов переменную 'variations' в данном скрипте

"""

from keras.models import load_model
from skimage import transform
from PIL import Image
import numpy as np
import os

# метки классов (keras использует алфавитный порядок для имён классов)
labels = {
    0: "dislike",
    1: "like",
}


# загрузка изображения и преобразование его в массив чисел
def load_image(img_file_path):
    np_image = Image.open(img_file_path)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (200, 200, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


# вывод результатов распознавания с цветовой пометкой
def check_expectations_and_print_result(expected_class, recognized_class):
    if expected_class == recognized_class:
        print("\033[36m {}".format(labels[recognized_class]))
    else:
        print("\033[31m {}".format(labels[recognized_class]))


if __name__ == "__main__":
    # загрузка модели
    model = load_model('true-love-model.h5')
    expected_image_class = 0

    variations = ["like", "dislike"]
    for variation in variations:

        # загрузка изображений и преобразование их в массивы чисел
        images_directory = "images-for-dataset/"+variation
        # images_directory = "generated-images-for-model/train/" + variation
        # images_directory = "generated-images-for-model/test/" + variation
        for file_name in os.listdir(images_directory):
            image_array = load_image(os.path.join(images_directory, file_name))

            # определение ожидаемых классов
            for key, value in labels.items():
                if value == variation:
                    expected_image_class = key

            # получение предсказаний модели
            recognized_image_class = int(model.predict_classes(image_array))

            # вывод результатов распознавания с цветовой пометкой
            check_expectations_and_print_result(expected_image_class, recognized_image_class)
