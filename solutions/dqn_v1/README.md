# DQN v1 — Deep Q-Network (Double DQN)

Полносвязная **Double DQN** для Mars Rover: replay buffer + target-сеть (soft update),
Huber-потеря, epsilon-greedy с линейным аннилированием. Модель без памяти (`M=1`),
Q-значения всех 31 команд подаются в выход `logits` (аргмакс на сервере выбирает команду).

## Особенности

- Контекст каждого решения: observation[160] + one-hot предыдущей команды + награда/флаг
  завершения/доля времени — признаки для сети те же, что у рекуррентного PPO.
- Одно RL-решение = 8 шагов физики (`frame_skip=8`), как при удержании команды в заезде.
- Replay buffer 100k переходов, batch 256, `gamma=0.99`, `tau=0.005`, эпсилон 1.0 → 0.05.
- Команда 18 («оба поршня») в action space есть, но физически не действует (`docs/exploits.md`).

## Запуск

```bash
python train.py                                   # полное обучение, экспорт в artifacts/policy.onnx
python train.py --total-frames 100000 --num-envs 8 \
  --capacity 20000 --batch-size 128               # короткий прогон для проверки
python check_policy.py artifacts/policy.onnx
```

## Пределы

- Без памяти сеть не видит рельеф дальше ~3 м (пассивный сенсор) и не знает фазы биомов;
  это сознательно «базовый» алгоритм для сравнения с PPO/A2C/REINFORCE.
- Для сильных результатов нужен рекуррентный вариант (см. `solutions/ppo_recurrent_v1`).
