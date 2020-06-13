"""
Данный скрипт производит генерацию файлов для создания датасета (как train, так и test выборок)

Для того, чтобы запустить его - достаточно закинуть в папку 'images-for-dataset' изображения с теми,
кто нравится внешне и кто нет (папки 'like' и 'dislike' соответственно).

Желательно использовать квадратные изображения в качестве исходных данных.

Данный скрипт можно использовать для генерации датасетов для любых классов в ваших конкретных случаях,
заменив папки 'like' и 'dislike' на другие, а также изменив в соответствии с названиями классов переменную 'variations'
в данном скрипте

"""
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image, ImageFile
import os


# создание квадратного изображения
def make_square(image):
    cols, rows = image.size

    if rows > cols:
        pad = (rows - cols) / 2
        image = image.crop((pad, 0, cols, cols))
    else:
        pad = (cols - rows) / 2
        image = image.crop((0, pad, rows, rows))

    return image


# генерирует случайным образом заданное количество изображений  и сохраняет их в указанную директорию
def generate_images(data_generator, generated_train_data_directory, converted_image, variation, stop_count):
    i = 0
    for _ in data_generator.flow(
            converted_image, batch_size=1,
            save_to_dir=generated_train_data_directory,
            save_prefix=variation + "-image", save_format="jpeg"):
        i += 1
        if i > stop_count:
            break


# генерация множества случайных изображений для создания датасета
def generate_images_for_dataset():
    image_width = 300
    image_height = 300

    # настройка генератора для изображений, входящих в датасет
    data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    variations = ["like", "dislike"]
    for variation in variations:

        # путь к изображениям, которые будут преобразованы
        raw_data_directory = "images-for-dataset/"+variation

        # пути к изображениям, которые будут использованы для подготовки данных для обучения нейросети
        generated_train_data_directory = "generated-images-for-model/train/"+variation
        if not os.path.exists(generated_train_data_directory):
            os.makedirs(generated_train_data_directory)

        generated_test_data_directory = "generated-images-for-model/test/" + variation
        if not os.path.exists(generated_test_data_directory):
            os.makedirs(generated_test_data_directory)

        # преобразование исходных изображений путём изменения размера и приведения их к квадратной форме
        for file_name in os.listdir(raw_data_directory):
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            image = Image.open(os.path.join(raw_data_directory, file_name))
            image.load()
            image = make_square(image)
            image = image.resize((image_width, image_height), Image.ANTIALIAS)
            image.save(os.path.join(generated_train_data_directory, file_name), 'JPEG', quality=90)

        # создание вариаций исходных изображений (здесь используются другие директории)
        for file_name in os.listdir(generated_train_data_directory):

            # преобразует изображение в массив с определенной структурой (3, 150, 150)
            converted_image = img_to_array(load_img(os.path.join(generated_train_data_directory, file_name)))

            # меняет структуру на (1, 3, 150, 150)
            converted_image = converted_image.reshape((1,) + converted_image.shape)

            # генерирует случайным образом заданное количество изображений и сохраняет их в указанную директорию
            generate_images(data_generator, generated_train_data_directory, converted_image, variation, 20)
            generate_images(data_generator, generated_test_data_directory, converted_image, variation, 5)


# TODO подключить нейросетку, которая распознаёт лица, чтобы обрезать фото невручную
if __name__ == '__main__':
    generate_images_for_dataset()
