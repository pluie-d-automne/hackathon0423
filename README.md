# hackathon0423
Student's hackathon. April 2023. Team: КотШредингера.

## Case Raleted Links
* [Garpix API](https://glsystem.net/dokumentaciya-k-api)
* [Garpix Case](https://docs.google.com/document/d/1OYgCF0F0AFQoFcH86BYoUUtqk_u1byTIsYUegyg6CUw/edit)
* [Questions Form](https://docs.google.com/spreadsheets/d/147NsgSBn5vw8UxskmBR5MTKUY0lrjTWpIp5C7yJsjOs/edit#gid=0)

## Hackaton CheckPoints
* <del>27.03 - Публикация задач</del>
* <del>10.04 - Чек-поинт 1 - Бизнес-корректировка (асинхронная обратная связь) - Надо подготовить презентацию и загрузить до 8:00 понедельника [сюда](https://drive.google.com/drive/folders/1Hg5C3nD-N6DY27QdaMsZGMZ3xL2q_qn9)</del>
* 21.04 - Чек-поинт 2 - Асинхронная предзащита (уже должны быть загружены ссылки на git)
* **28.04 - Кодфриз с 20:00**
*30.04 - Закрытие. Выступление лучших команд.

## Team: КотШредингера
* Георгий Шипигузов @George2301
* Елизавета Алмазова @lizalmazova
* Михаил Губанов @yainformal
* Полина Кудрявцева @pluie-d-automne
* Сергей Евстратов @esvesv
* Сергей Мартынюк @martynyuks 

## Q&A

**Данные в датасете - это единоразовая выгрузка, или  API будет туда периодически подкладывать новые файлы? Нам нужно использовать именно эти данные из архива, не нужно подключаться к API?**\
Данные использовать именно эти. Ссылку на api дал чисто в ознакомительном порядке. Естественно, с учетом типа грузового пространства

**Нам необходимо прогнозировать качество укладки только на основании размеров параллелепипеда (“длина”, “ширина”, “высота”) или нужно самостоятельно определить перечень параметров на основе предоставленных данных?**\
Предполагаем следующие характеристики: размеры (длина, высота, щирина), возможность штабелирования (ограничение на количество слоев укладки), возможность кантования (переворачивать в процессе погрузки)

**Что будет передаваться API на входе - некий массив с габаритами коробок в наборе? Или тут тоже мы сами определяем из набора доступных в датасете  метрик, что и в каком виде передавать?**\
Наверное, в идеале подавать именно такие файлы как в датасете, но это не сильно принципиально.

**Правильно ли мы понимаем постановку задачи: что мы берём заданные достигнутые плотности и учимся их предсказывать. Не пытаемся улучшить достигнутый укладчиком результат?**\
Да. Основная задача предсказать плотность укладки. `Дополнительно - это было бы важно (но в качестве требования не указано) - получить некоторые инварианты характеристик совокупности грузов, которые влияют на плоность укладки.`

**В чем отличие метрик density_percent и filling_space_percent?**\
Это правильный вопрос :)\
`density_percent` - расчитывается как доля заполненного грузвого пространства, ограниченного плоскостью, к которой относится самая верхняя грань самого верхнего груза;\
`filling_space_percent` - доля заполненного грузового пространства, объем которого расчитан по масимальным габаритам.
Поэтому в общем случае density_percent больше filling_space_percent. Ценность представляет повышение filling_space_percent, но density_percent показывает возможно достижимое качество укладки груза. Мы полагаем, что при наличии производительного алгоритма можно переспрделять грузы между грузовыми пространствами (например, паллетами внутри одной фуры), так чтобы filling_space_percent увеличивался.

## Feedback
* **Checkpoint 1**\
В представленном первичном решении отражена вся последовательность и логика решения. Т.е. команда имеет реалистичную гипотезу для решения. Проведен анализ первичных данных и дана их качества и количества. В том числе выдвинуто предположение о том, что количества наблюдений в датасете может быть недостаточно для полноценного обучения нейронной сети. 
Предложены три различных подхода к решению задачи, включая описание способов их реализации. Все три подхода представляются реалистичными.\
При необходимости заказчик готов увеличить размер датасета, а также предоставить доступ к интерфейсу системы оптимального планирования размещения грузов GLS
