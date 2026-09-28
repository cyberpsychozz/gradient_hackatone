# Реестр сделанных решений (solutions_made)

Отслеживание реализованных решений в `solutions/`. Каждое решение — самостоятельная посылка
(Dockerfile + train.py + README.md в корне), обучающая политику для Mars Rover и экспортирующая
`/output/policy.onnx` по контракту `run_rules.md`.

## Решения

| # | Папка | Алгоритм | Память (M) | Статус | Проверка | Когда |
|---|---|---|---|---|---|---|
| 1 | `solutions/ppo_recurrent_v1` | PPO (рекуррентный, GRU) | 128 | ✅ сделан | `python check_policy.py artifacts/policy.onnx` | 2026-09-29 |
| 2 | `solutions/dqn_v1` | Double DQN (replay buffer + target net) | 1 | ✅ сделан | те же smoke-тесты | 2026-09-29 |
| 3 | `solutions/a2c_v1` | A2C (синхронный actor-critic, n-step) | 1 | ✅ сделан | те же smoke-тесты | 2026-09-29 |
| 4 | `solutions/reinforce_v1` | REINFORCE с baseline (MC returns) | 1 | ✅ сделан | те же smoke-тесты | 2026-09-29 |
| 5 | `solutions/cem_v1` | CEM (эволюция линейной политики) | 1 | ✅ сделан | те же smoke-тесты | 2026-09-29 |

## Как добавить решение

1. Новая папка `solutions/<algorithm>_v<n>/` с контрактом посылки: `Dockerfile`, `train.py`,
   `README.md` в корне; `model.py`, `check_policy.py`, `tests/test_submission_smoke.py` рядом.
2. Модель обязана реализовывать `forward(observation, previous_action, previous_reward,
   previous_done, trial_progress, trial_start, memory) -> (logits, next_memory)`.
3. Добавить строку в таблицу выше и прогнать smoke-тесты.

## Заметки

- `ppo_recurrent_v1` дополнительно содержит источники среды (`cpp/`, `python/`, `setup.py`) и
  переустанавливает пакет в образе — это стартовая посылка организаторов, поверх которой написан
  независимый PPO.
- Остальные решения — «лёгкие»: они не копируют источники среды, т.к. `arena-base` уже содержит
  `mars_rover_env 0.16.0`; Dockerfile только копирует файлы и гоняет smoke-тесты.
- Метрика качества — медиана максимальной дистанции на фиксированных сидах (`docs/run_rules.md`),
  а не суммарная награда обучения.
