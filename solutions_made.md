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

## Локальная проверка (2026-09-29)

Прогнано на CPU (venv: torch 2.14+cpu, onnx 1.23, onnxruntime 1.30, native `.so` из `environment/`):

- Все smoke-тесты (`export + validate_policy` с onnxruntime) проходят для всех решений.
- Найдено и исправлено при локальном прогоне:
  - torch.onnx.export выбрасывает неиспользуемый вход `trial_start` → добавлен
    `ensure_contract_inputs()` в fallback-экспорт (dqn/a2c/reinforce/cem);
  - dqn_v1: `ReplayBuffer.push` получал батч вместо одной записи;
  - cem_v1: кандидаты в float64, тензоры требуют float32;
  - reinforce_v1: `KeyError 'reward'` → `len(episode["returns"])`.
- Короткие заезды обучения (8 сред, ~2M фреймов): DQN медиана 170–210 м, REINFORCE ~60–70 м,
  CEM best ~160 м (на частичных заездах). Это sanity-check, не итоговая оценка.
