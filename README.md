# Telegram Training Bot – Frontend Gateway

Этот репозиторий содержит только «фронт» для телеграм‑тренажёра: бот принимает апдейты, качает voice, делает транскрибацию (если настроено), нормализует событие и отправляет его в AI‑бэкенд, а затем возвращает пользователю ровно то, что вернул бэкенд (текст + клавиатура).

## Быстрый старт

1. Подготовьте Python 3.10+ и установите зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Заполните `.env` на основе примера:
   ```bash
   cp .env.example .env
   # отредактируйте токены/URL
   ```
3. Выберите режим:
   - Webhook: `USE_WEBHOOK=true`, пропишите внешний адрес (`WEBHOOK_URL` с путём `WEBHOOK_PATH`), настроьте домен/туннель.
   - Long polling: `USE_WEBHOOK=false`, `WEBHOOK_URL` не нужен.
4. Запустите бота:
   ```bash
   python main.py
   ```
   - В webhook-режиме бот сам выставит webhook на `WEBHOOK_URL`, слушает `LISTEN_HOST:LISTEN_PORT + WEBHOOK_PATH`. Healthcheck: `GET /health`.
   - В polling-режиме бот просто стартует long polling и не поднимает HTTP-сервер.

## Конфигурация

Все переменные читаются из окружения или `.env`:
- `BOT_TOKEN` — токен Telegram Bot API (обязательно).
- `USE_WEBHOOK` — `true` (по умолчанию) для вебхука, `false` — для long polling.
- `WEBHOOK_URL` — публичный URL вебхука (обязательно в webhook-режиме), например `https://your-domain/tg/webhook`. Путь в URL должен совпадать с `WEBHOOK_PATH`.
- `WEBHOOK_PATH` — локальный путь, который слушает сервер (по умолчанию `/tg/webhook`), должен совпадать с путём в `WEBHOOK_URL`.
- `WEBHOOK_SECRET` — секрет для заголовка `X-Telegram-Bot-Api-Secret-Token` (настроить и здесь, и в Telegram).
- `LISTEN_HOST` / `LISTEN_PORT` — где слушать HTTP-сервер.
- `BACKEND_URL` — базовый URL AI-бэкенда, например `https://api.example.com`.
- `BACKEND_TOKEN` — необязательный Bearer-токен, если бэкенд требует авторизации.
- `PUSH_TOKEN` — опциональный Bearer-токен, который бэк присылает в `Authorization` для /push.
- `TRANSCRIBE_URL` — URL сервиса транскрибации; если не задан, бот отправит в бэкенд placeholder текста.
- `TRANSCRIBE_TOKEN` — опциональный Bearer‑токен для транскрибации.
- `OPENROUTER_API_KEY` — ключ OpenRouter; если задан вместе с `OPENROUTER_ASR_MODEL`, для voice используется OpenRouter ASR.
- `OPENROUTER_ASR_MODEL` — идентификатор модели для транскрибации на OpenRouter (например `openai/whisper-large-v3`).
- `OPENROUTER_ASR_URL` — endpoint для аудио транскрибации на OpenRouter (по умолчанию `https://openrouter.ai/api/v1/audio/transcriptions`).
- `DEFAULT_REPLY_TEXT` — текст по умолчанию при старте (иначе используется встроенный приветственный).
- `REQUEST_TIMEOUT_SECONDS` / `ASR_TIMEOUT_SECONDS` — таймауты для бэкенда и ASR.
- `FFMPEG_BINARY` — путь к ffmpeg для конвертации voice → wav; если не найден, отправляется оригинальный OGG.
- `MAX_TG_MESSAGE_LENGTH` — лимит для нарезки длинных ответов (по умолчанию 3900 символов; Telegram максимум 4096).

## Контракт с бэкендом

### Куда шлёт фронт
POST `{BACKEND_URL}/ingest`

Пример запроса:
```json
{
  "event_id": "uuid-123",
  "telegram_update_id": 987654321,
  "user": { "user_id": 111, "chat_id": 222, "username": "andrey" },
  "event": { "type": "text", "text": "Мой ответ на кейс..." },
  "meta": { "source": "telegram", "client_ts": "2025-12-29T10:15:00+01:00" }
}
```

Callback:
```json
{
  "event_id": "uuid-124",
  "telegram_update_id": 987654322,
  "user": { "user_id": 111, "chat_id": 222, "username": "andrey" },
  "event": { "type": "callback", "data": "case:next" },
  "meta": { "source": "telegram" }
}
```

Voice (ASR уже на фронте):
```json
{
  "event_id": "uuid-125",
  "telegram_update_id": 987654323,
  "user": { "user_id": 111, "chat_id": 222, "username": "andrey" },
  "event": { "type": "text", "text": "текст после ASR", "source": "voice" },
  "meta": {
    "source": "telegram",
    "asr": { "confidence": 0.86 },
    "voice_seconds": 3
  }
}
```

Ответ бэкенда (рекомендуемый):
```json
{
  "actions": [
    {
      "type": "send_message",
      "chat_id": 222,
      "text": "Обратная связь по ответу...\n\nДавай попробуем ещё раз.",
      "parse_mode": "HTML",
      "keyboard": {
        "inline": [
          [{"text": "Повторить", "data": "case:retry"}],
          [{"text": "Дальше", "data": "case:next"}]
        ]
      }
    }
  ]
}
```

Если `actions` нет, поддерживается старый формат: `text` + `keyboard`.

### Что принимает фронт для напоминаний
POST `/push` (опциональный `Authorization: Bearer ${PUSH_TOKEN}`)
```json
{
  "actions": [
    {
      "type": "send_message",
      "chat_id": 222,
      "text": "Напоминаю: незавершённый кейс. Продолжаем?",
      "keyboard": {
        "inline": [
          [{"text": "Продолжить", "data": "resume:yes"}],
          [{"text": "Напомнить позже", "data": "remind:later"}]
        ]
      }
    }
  ]
}
```

## Что умеет бот
- `/start` отправляет приветствие и (если есть) клавиатуру из бэкенда.
- Текстовые сообщения: сразу уходят в бэкенд.
- Voice: бот скачивает файл, при наличии `TRANSCRIBE_URL` отправляет его на транскрибацию и передаёт транскрипт в бэкенд; иначе шлёт placeholder `"[voice message]"`.
- Если транскрибация упала — просит повторить голосом или отправить текст.
- Callback‑кнопки: передаются как `callback`‑события.
- Все ошибки бэкенда/транскрибации логируются; пользователю показывается мягкое сообщение об ошибке.
- Дедупликация `update_id` — повторные апдейты сразу отбрасываются.
- Вебхук защищён секретом `WEBHOOK_SECRET` через заголовок `X-Telegram-Bot-Api-Secret-Token`.
- Форматирование: бот рендерит HTML (ParseMode.HTML). Закрывайте теги, не кладите нестандартизированный Markdown. Бэкенд желательно держать ответы < 3500–3800 символов; бот режет длинные тексты по параграфам в несколько сообщений при превышении лимита Telegram (4096).
- Команды/кнопки фронта: `/start` (приветствие + меню), `/menu` (меню), `/diagnostic` (старт диагностики). Inline-кнопки по умолчанию: начать диагностику, перейти к тренажёру, прогресс, напоминания, выбор навыка (обратная связь/ИПР), оглавление. Эти кнопки шлют `action:*` в бэкенд, который должен решить дальнейший сценарий (опросник, интерпретация, кейсы, напоминания, прогресс).

## Локальная разработка
- Запуск с hot reload: `python -m watchfiles main.run` (нужно `watchfiles`, не входит в requirements).
- Логи в stdout; уровни настраиваются через `LOG_LEVEL` (`INFO` по умолчанию).

## Структура
- `main.py` — входная точка, создание бота и регистрация хендлеров.
- `config.py` — загрузка настроек из окружения.
- `backend_client.py` — отправка событий в AI‑бэкенд.
- `transcriber.py` — скачивание voice и отправка в сервис транскрибации.
- `keyboard.py` — сборка inline‑клавиатуры из ответа бэкенда.
- `backend/` — минимальный бэкенд на FastAPI с эндпоинтом `/ingest` и AI‑интеграцией (OpenRouter опционально).

## Бэкенд (пример)
- Завести venv и поставить зависимости:
  ```bash
  cd backend
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
  ```
- Настройки через переменные:
  - `BACKEND_HOST` / `BACKEND_PORT` — адрес/порт бэкенда.
  - `FRONTEND_PUSH_URL` — URL фронта для `/push` (например `http://localhost:8080/push`), чтобы слать напоминания.
  - `FRONTEND_PUSH_TOKEN` — токен для `Authorization: Bearer ...` если фронт закрыт.
  - `REMINDER_DELAY_SECONDS` — задержка перед напоминанием.
  - AI: `OPENROUTER_API_KEY` (если задан, ответы кейсов оцениваются моделью), `OPENROUTER_MODEL` (по умолчанию `gpt-3.5-turbo`), `OPENROUTER_BASE_URL`, `OPENROUTER_TEMPERATURE`.
- Контракт:
  - `POST /ingest` — принимает события от фронта (см. пример выше), возвращает `actions` для отправки в Telegram.
  - `POST /push` (на фронте) — бэкенд может дернуть, чтобы отправить напоминание.
  - `GET /health`, `GET /metrics` — служебные.

## Тестовый прогон
- Убедитесь, что токен бота корректен (в т.ч. разрешения на voice).
- Запустите `python main.py`, отправьте в бот текст/voice, посмотрите логи.
- При необходимости поднимите мок бэкенда (например, `uvicorn mock:app`) — контракт выше.
