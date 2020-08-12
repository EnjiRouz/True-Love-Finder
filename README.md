# True-Love-Finder
Используя данный проект, вам больше не придется множество раз делать swipe в приложениях по типу Tinder и Badu. Вы сможете обучить нейросеть в соответствии со своими предпочтениями, а далее интегрировать её для связки с нужным вам приложением. Нейросеть будет подсказывать вам, понравится вам конкретный партнёр или нет.

> С помощью написанных мной скриптов вы сможете реализовать проекты для личного пользования, где вам требуется сгенерировать датасеты, разделённые на классы категорий, а затем обучить и протестировать на них нейронную сеть.



### Содержимое проекта
Для тренировки нейронной сети подготовлено несколько скриптов на Python:
| Название файла со скриптом | Назначение|
| -------------------------- | ----------|
| `image_generator.py` | генерация случайных изображений на основе загруженных в папку `images-for-dataset` фотографий с теми людьми, кто вам нравится внешне и кто нет (папки `like` и `dislike` соответственно) |
| `face_detector.py`  | поиск лица на изображениях при первичной обработке (самостоятельно запускать его не требуется, поскольку файл включён в работу `image_generator.py`)|
| `CNN.py` | создание и тестирование на материале обучения модели нейронной сети на основе собранного с помощью `image_generator.py` датасета (используется как `train`, так и `test` выборка)|
| `CNN_test.py` | проверка предсказаний нейронной сети на основе валидационной выборке из папки `validation-images`|

Данный проект **можно использовать для генерации датасетов для ЛЮБЫХ классов** в ваших конкретных случаях, заменив папки `like` и `dislike` на другие (подходящие под ваши задачи, например "cats" и "dogs"), а также изменив в соответствии с названиями классов переменную `variations` в скриптах проекта

**Если вашему проекту не нужно распознавать лица** - просто уберите из кода ту часть, которая отвечает за поиск лица на изображении и последующую обрезку либо замените её на другой фрагмент с распознаванием, подходящим под ваши задачи.

### Пример использования
Я взяла изображения с двумя персонажами игры Detriot: Become Human и поместила их в папку `images-for-dataset`. 
Изображения с Коннором, как с более привлекательным для меня персонажем, я добавила в папку `like`.
Изображения с Маркусом и Саймоном, которые далеки от моего типажа в плане внешности, я поместила в папку `dislike`:

![Screenshot_4](https://user-images.githubusercontent.com/26218291/90047669-b7f60380-dceb-11ea-9790-2ddc443809f8.png)

После того, как в папку было добавлено большое количество изображений - можно начинать генерацию датасета, запустив файл `image_generator.py`.
Это может занять до нескольких минут (каюсь, нужно оптимизировать код, на что в ночное время меня уже не хватает :smile:)

После вывода в консоли надписи **`Done!`** можно приступать к обучению нейронной сети. Для этого потребуется запустить файл `CNN.py`. 


> Не забудьте поставить [CUDA](https://developer.nvidia.com/cuda-downloads) от NVIDIA для быстрого и комфортного обучения с помощью видеокарты.


После завершения обучения в консоли будет выведено сообщение **`Model is successfully stored and tested!`**, а в папке появится тяжеловесный файл обученной модели нейронной сети под названием `true-love-model.h5`. С результатами обучения можно ознакомиться в файле `model_summary.txt`. Если вас не устраивает результат - вы всегда можете скорректировать параметры под себя, учитывая размеры изображений, желаемое количество карт признаков, архитектуру нейронной сети и т.д.

Для тестирования на валидационной выборке достаточно запустить `CNN_test.py`, после чего можно увидеть успешные и ошибочные срабатывания нейросети, показанные цветовой пометкой в консоли (*красный цвет - значит ошибочное срабатывание*). Изображения для валидационнной выборки были отобраны на момент генерации датасета и не входят в выборку, на которой проводилось обучение.