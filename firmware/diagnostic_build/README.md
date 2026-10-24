# ESP32 GRU firmware for protocol `gru-esp32-v1`

Это адаптированная версия `KamererM/gru_predictor_esp` для воспроизводимого leakage-free перезапуска эксперимента. Исходная база: commit `ad254ab8d8901a8adc14ffa2a82648560ad7da62`.

## Что изменено

- Добавлен `GET /metadata` на порту 8080.
- `POST /data` принимает `run_id`, строковый `timestamp_ns`, стабильный `request_id`, `size=32` и 32 raw hourly kWh values.
- Ответ возвращает идентификаторы, `prediction`, `execution_time_us`, точные `input_int8[32]` и `output_int8`.
- Последние 32 ответа кэшируются по `request_id`. Повтор с теми же timestamp и входом возвращает тот же ответ без повторного inference; конфликт идентификатора отклоняется.
- Нормализация использует training-only float32 constants; квантование выполняет `roundf` и clipping в `[-128, 127]`.
- На старте проверяются shape `[1,32,1]`, INT8 I/O и quantization parameters модели.
- Metadata связывает устройство с run, моделью, generated sources, scaler, `sdkconfig`, resolver source, вычисляемым build ID и ESP-IDF `app_elf_sha256`; flash, PSRAM и chip revision читаются с устройства.
- SSID и пароль удалены из исходников и задаются через menuconfig.

## Предварительные условия

- ESP-IDF 6.0.x; reference build uses 6.0.3.
- `espressif/esp-tflite-micro` 1.4.1.
- Выполненный leakage-free rerun, создавший `run_manifest.json`, `model_data.cc`, `model_data.h`, `preprocessing_constants.h`, `test_windows.npz` и `golden_vectors.npz`.

Репозиторий содержит безопасный `sdkconfig.defaults`. Локальный `sdkconfig`, создаваемый ESP-IDF, исключён через `.gitignore`.
`dependencies.lock` создаётся reference build под ESP-IDF 6.0.3 и должен сохраняться вместе с артефактами эксперимента.

## Сборка и прошивка

Сначала настройте сеть. Не публикуйте получившийся `sdkconfig`, поскольку он содержит Wi-Fi credentials.

```bash
idf.py menuconfig
```

В меню `GRU predictor network settings` задайте SSID и пароль. Там же задаётся tensor-arena capacity и минимальный остаток heap; инициализация прерывается, если после `AllocateTensors()` остаётся меньше заданного резерва. Затем установите артефакты конкретного run и создайте embedded metadata. Build ID вычисляется автоматически из зафиксированных deployment inputs.

```bash
python tools/generate_deployment_metadata.py \
  --results-dir /absolute/path/to/results_primary \
  --firmware-root . \
  --firmware-commit <git-commit> \
  --board <exact-board-or-module>

idf.py fullclean
idf.py build
```

Если каталог является корнем Git checkout, параметр `--firmware-commit` можно опустить: скрипт получает полный SHA через `git rev-parse`. При сборке из архива укажите SHA commit, в который вы предварительно внесли эти изменения.

После сборки повторите ту же команду с app image. Это добавляет локальный SHA-256, размер image и извлечённый из него `app_elf_sha256` в `device_metadata_static.json`. Генератор обновляет source/header только при изменении содержимого, поэтому post-build вызов не меняет их timestamps и не должен запускать Ninja rebuild.

```bash
python tools/generate_deployment_metadata.py \
  --results-dir /absolute/path/to/results_primary \
  --firmware-root . \
  --firmware-bin build/gru_predictor.bin \
  --firmware-commit <git-commit> \
  --board <exact-board-or-module>

idf.py -p /dev/ttyUSB0 flash monitor
```

Проверьте в выводе `idf.py flash`, что сборка сообщает отсутствие работы, и что прошивается уже хэшированный `build/gru_predictor.bin`. Если Ninja всё же пересобрал app image, не запускайте evaluation: повторите post-build metadata generation для нового binary, затем снова выполните flash. Если исходники или `sdkconfig` изменились, повторите процедуру с `fullclean`.

## Захват metadata

После подключения устройства:

```bash
python tools/capture_device_metadata.py \
  --url http://ESP32_IP:8080/metadata \
  --static device_metadata_static.json \
  --output device_metadata.json
```

Утилита не доверяет ручному описанию: она сравнивает live значения с build metadata и сохраняет объединённую запись с SHA-256 конкретного app image.

## Проверка golden vectors и test set

Запускайте evaluator из leakage-free rerun package:

```bash
python evaluate_esp32.py \
  --results-dir /absolute/path/to/results_primary \
  --url http://ESP32_IP:8080/data \
  --metadata-url http://ESP32_IP:8080/metadata \
  --device-metadata /absolute/path/to/device_metadata.json \
  --firmware-binary /absolute/path/to/build/gru_predictor.bin \
  --sdkconfig /absolute/path/to/sdkconfig \
  --resolver-source /absolute/path/to/main/gru_predictor.cc \
  --warmup 20 \
  --bootstrap-resamples 10000
```

Оценка прекращается при несовпадении identity, timestamps, target pairing, model/scaler hashes, INT8 входов или INT8 выходов.

## REST contract

Пример запроса:

```json
{
  "run_id": "run UUID",
  "timestamp_ns": "1167609600000000000",
  "request_id": "run UUID:test:0:1167609600000000000",
  "size": 32,
  "data": [0.1, 0.2, 0.3],
  "diagnostic": true
}
```

Поле `data` в реальном запросе должно содержать ровно 32 числа. `timestamp_ns` является строкой, чтобы избежать потери точности в JSON.

Пример ответа:

```json
{
  "run_id": "run UUID",
  "timestamp_ns": "1167609600000000000",
  "request_id": "run UUID:test:0:1167609600000000000",
  "prediction": 0.41,
  "execution_time_us": 1234,
  "input_int8": [-10, -9],
  "output_int8": -4
}
```

Массив `input_int8` в реальном ответе содержит 32 элемента.

## Ограничения текущей проверки

В этой среде ESP-IDF toolchain отсутствует, поэтому выполнены только Python tests и статическая проверка C++/protocol consistency. Publication readiness требует локальной сборки, прошивки физического ESP32, успешного metadata handshake, точного golden-vector test и полного held-out evaluation.

После первой успешной сборки сохраните созданный `dependencies.lock` вместе с экспериментальными артефактами. Он фиксирует фактически разрешённые транзитивные зависимости.

В исходной публичной версии Wi-Fi credentials были записаны непосредственно в `wifi.cc`. В этой версии они удалены. Если те credentials были реальными, их следует заменить, поскольку удаление из новой копии не удаляет их из истории исходного репозитория. Не публикуйте app binary: Wi-Fi credentials неизбежно входят в локально собранный образ.
