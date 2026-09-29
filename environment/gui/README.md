# Student GUI

GUI использует только студенческий каталог: встроенные и train-биомы. Без явной
конфигурации среда запускается с train-split. Если передан `--config`, GUI
сохраняет указанный в нём публичный набор биомов и параметры сложной трассы.

```bash
make build
make play
make play ARGS="--list-biomes"
make play ARGS="--biome gravity_shelf_lug --debug"
```

## Просмотр обученной политики

Установите `onnxruntime` в то же окружение, где запускается GUI, затем передайте
экспортированный ONNX-файл:

```bash
python -m pip install onnxruntime
mars-rover-play --policy ../../solutions/a2c_v1/artifacts/policy.onnx --seed 42 --debug
```

Путь в примере рассчитан на запуск из `environment/gui`; можно использовать
абсолютный путь. GUI выбирает `argmax(logits)` раз в 8 физических кадров и
передаёт политике накопленную награду и память. `R` повторяет тот же мир,
`T` создаёт новый. После окончания заезда симуляция останавливается.

Для PPO сначала требуется обучение: в `solutions/ppo_recurrent_v1/artifacts`
сейчас нет `policy.onnx`. Публичные миры GUI не совпадают с закрытой оценкой.

## Сложные локальные трассы

Из каталога `environment/gui` установите обновлённую студенческую среду
и запустите заезд на плотной трассе:

```bash
python -m pip install --no-build-isolation --no-deps ../../solutions/rover_best_v1
make build
make play ARGS="--config ../../solutions/rover_best_v1/python/mars_rover_env/configs/eval_stress_dense.yaml --policy ../../solutions/rover_best_v1/artifacts/policy.onnx --seed 904 --debug"
```

Для погодного профиля замените `eval_stress_dense.yaml` на
`eval_stress_weather.yaml`. Это локальные стресс-миры, а не точные трассы
закрытой платформы.
