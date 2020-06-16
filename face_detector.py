"""
Данный скрипт производит поиск ОДНОГО лица (если нужно больше - нужно убрать break в цикле)
на изображении и его обрезку с последующим сохранением,
а также для отладки присутствует метод для вывода изображения в отдельном окне (но он не используется по умолчанию)

"""

import cv2

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# просмотр изображения в новом окне (вообще это можно убрать, но для дебага удобно)
def view_image(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# определение положения лица на изображении и обрезка в соответствии с границами лица с последующим сохранением
def detect_face_and_crop_image(path_to_image, final_file_name):
    image = cv2.imread(path_to_image)

    # если цвет неважен - можно использовать gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=1, minSize=(300, 300))

    # обрезание картинки в точке определения лица
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            image = image[y:y + h, x:x + w]

            # для отладки можно использовать:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # view_image(image, "face_detected")
            break

        # сохранение изображения
        cv2.imwrite(final_file_name, image)


# определение положения лица на изображении и обрезка в соответствии с границами лица
def detect_face(path_to_image):
    image = cv2.imread(path_to_image)

    # если цвет неважен - можно использовать gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=1, minSize=(300, 300))

    # обрезание картинки в точке определения лица
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            return image[y:y + h, x:x + w]
