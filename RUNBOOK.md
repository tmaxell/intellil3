# Runbook: Воспроизведение экспериментального контура IntelliL3

Краткое руководство для запуска полного набора экспериментов.
Все команды выполняются из корня репозитория с активированным виртуальным окружением.

```bash
source .venv/bin/activate
```

---

## 1. Smoke-тесты (быстрая проверка работоспособности)

**Measurement smoke** — один прогон синтетической нагрузки, ~3 с:

```bash
bash scripts/run_smoke_measurement.sh
# Артефакты: benchmarks/results/smoke_measurement/
```

**Comparison smoke** — сравнение двух систем на минимальной нагрузке, ~3 с:

```bash
bash scripts/run_smoke_comparison.sh
# Артефакты: benchmarks/results/smoke_comparison/
```

Оба smoke-скрипта пишут `run_summary.json` и `run_summary.md` рядом с остальными артефактами.

---

## 2. Lightweight-прогон (все 10 экспериментов, ~1 мин)

Запускает все конфиги из `benchmarks/configs/lightweight/`, 3 повторения каждый:

```bash
bash scripts/run_lightweight_experiments.sh
# Артефакты: benchmarks/results/lightweight/
#   <exp_name>/run_summary.json   — сводка по эксперименту
#   <exp_name>/comparison.json    — детальное сравнение
#   comparison_index.csv/.md      — сводный индекс
```

Параметры переопределяются через переменные окружения:

```bash
REPETITIONS=1 OUTPUT_DIR=/tmp/lw bash scripts/run_lightweight_experiments.sh
```

---

## 3. Storage experiments (exp01–exp04)

Оценивает политики кэширования и вытеснения на типовых LLM-нагрузках:

```bash
bash scripts/run_storage_experiments.sh
# Артефакты: benchmarks/results/storage/
# Покрывает: exp01_long_context, exp02_rag_heavy, exp03_multi_user, exp04_policy_comparison
```

---

## 4. Agentic experiments (exp05–exp09)

Оценивает agentic-системы на многошаговых workflow-нагрузках, включая retail-домен:

```bash
bash scripts/run_agentic_experiments.sh
# Артефакты: benchmarks/results/agentic/
# Покрывает: exp05–exp08 (agentic ReAct/multi-agent/tool/branching) + exp09 (retail workflow)
```

---

## 5. RAG experiments (exp10+)

Оценивает системы на реалистичной RAG-нагрузке с evidence-grounded метриками:

```bash
bash scripts/run_rag_experiments.sh
# Артефакты: benchmarks/results/rag/
# Покрывает: exp10_rag_realistic (и будущие exp1x_*.yaml)
```

---

## 6. Полный прогон всех экспериментов

```bash
bash scripts/run_all_experiments.sh
# Артефакты: benchmarks/results/
```

Параметры по умолчанию: `REPETITIONS=5`. Для ускорения:

```bash
REPETITIONS=3 bash scripts/run_all_experiments.sh
```

---

## 7. Optional: LLM sanity-check

Проверяет, что workload-паттерны (shared_prefix, no_reuse, workflow_like)  
соответствуют ожидаемым характеристикам KV-кэша — без обязательного LLM.

**Dry-run (без LLM, ~1 с):**

```bash
bash scripts/run_llm_sanity_check.sh dry_run benchmarks/results/sanity
# Артефакты: benchmarks/results/sanity/sanity_dry_run_<ts>.json/.md + run_summary.json
```

**С Ollama (требует запущенного `ollama serve`):**

```bash
bash scripts/run_llm_sanity_check.sh ollama benchmarks/results/sanity llama3.2:1b
```

Отсутствие Ollama не ломает основной pipeline — все остальные команды работают независимо.

---

## 8. Сводка по каждому прогону

После каждого `run_comparison` / `MeasurementSuite.run` / sanity-check автоматически  
создаётся пара артефактов в той же директории:

| Файл | Содержимое |
|---|---|
| `run_summary.json` | Ключевые метрики, verdict (✓/✗/—), интерпретация |
| `run_summary.md` | Читаемая таблица, пригодная для переноса в ВКР |

Пример быстрого просмотра сводки:

```bash
cat benchmarks/results/lightweight/exp10_rag_realistic_lightweight/run_summary.md
```

---

## Структура артефактов одного эксперимента

```
benchmarks/results/<mode>/<exp_name>/
├── raw_runs.json          # сырые результаты всех повторений
├── summary.json           # агрегированные метрики по системам
├── measurement_report.md  # отчёт measurement-слоя
├── comparison.json        # детальное попарное сравнение
├── comparison.csv         # то же в CSV
├── comparison_report.md   # читаемый отчёт сравнения
├── run_summary.json       # ← компактная сводка (этап 7)
└── run_summary.md         # ← читаемая сводка для ВКР (этап 7)
```

---

## Воспроизведение конкретного эксперимента вручную

```bash
# Measurement только
python -m benchmarks.measurement.run_measurement \
  --config benchmarks/configs/exp10_rag_realistic.yaml \
  --output-dir benchmarks/results/exp10 \
  --repetitions 3

# Comparison (baseline vs candidate)
python -m benchmarks.measurement.run_comparison \
  --config benchmarks/configs/exp10_rag_realistic.yaml \
  --output-dir benchmarks/results/exp10 \
  --repetitions 3

# Все эксперименты по glob
python -m benchmarks.measurement.run_all_comparisons \
  --config-glob "benchmarks/configs/exp1[0-9]_*.yaml" \
  --output-dir benchmarks/results/rag \
  --repetitions 3
```

---

## Быстрая проверка корректности после изменений

```bash
# Полный тест-сьют (~6 с)
python -m pytest tests/ -q

# Только RAG-тесты
python -m pytest tests/test_rag_realistic_benchmark.py tests/test_rag_realistic_workload.py -q

# Только сводка
python -m pytest tests/test_run_summary.py -q
```
