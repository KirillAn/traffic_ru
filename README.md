# Распознавание и классификация дорожных знаков

## Цель проекта:

Создать прототип помощника для водителей, который будет оповещать их о дорожных знаках. **Код в проекте приведен к формату pep8 с помощью black**

### DoD:
Помощник должен уметь точно и быстро распознавать дорожные знаки в различных условиях освещения и погоды, информируя водителя об ограничениях скорости, запретах, предупреждениях и других важных указаниях на дороге.


## Датасет

Для релизации модели использоваля [Russian traffic sign images dataset](https://www.kaggle.com/datasets/watchman/rtsd-dataset) состоящий из изображений российских доорожных знаков, которые были сняты с помощью видеорегистратора


## Модели

Для реализации проекта была выбрана [YOLOv8](https://github.com/ultralytics/ultralytics)

### Тестирование моделей:

**YOLOv8 Nano (YOLOv8n)**

F1 score

![F1_curve](https://github.com/KirillAn/traffic_ru/assets/69241093/08ab8177-327c-42e7-ae8c-c11bb2486de0)

Loss+metrics

![results](https://github.com/KirillAn/traffic_ru/assets/69241093/0d094644-643a-4cff-8fbc-0b2619bce1e1)

**YOLOv8s**

F1 score

![F1_curve](https://github.com/KirillAn/traffic_ru/assets/69241093/09fd20ba-d363-4fcb-b87c-82e598d07dd0)

Loss+metrics

![results](https://github.com/KirillAn/traffic_ru/assets/69241093/0b5f65c7-0160-4bbf-b399-aee1dd551eb0)


### Итоги

По результатам тестов остановились на модели YOLOv8n с использованием 60 эпох, как наиболее оптимального значения.
YOLOv8n использовалась и для классификации знаков и для их детекции.



## Тестирование

[Видео для теста модели](https://drive.google.com/drive/u/0/folders/1J8YER5fkejEVTa6xXUT6CQa-dITuXh1r)

