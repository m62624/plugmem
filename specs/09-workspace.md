# 09 — структура workspace и наследование Cargo-полей

> Каркас создан и проверен (`cargo check --workspace` зелёный; arena/core
> собираются под `wasm32v1-none --no-default-features`). Паттерн — зеркало
> воркспейса elenchus, без CI-части (она добавится на этапе 5 вместе с
> cargo-dist).

## Дерево

```
plugmem/
├── Cargo.toml               ← [workspace]: members, workspace.package, workspace.dependencies
├── Cargo.lock               ← коммитится (в workspace есть бинарники)
├── crates/
│   ├── plugmem-arena/       ← no_std lib
│   ├── plugmem-core/        ← no_std lib
│   ├── plugmem-host/        ← std lib
│   ├── plugmem-cli/         ← bin "plugmem"
│   ├── plugmem-mcp/         ← bin "plugmem-mcp"
│   ├── plugmem-napi/        ← cdylib+rlib, publish = false (нативный Node-аддон, уходит в npm)
│   └── plugmem-testgen/     ← std lib, publish = false (внутренний)
├── reference/opaque-v1/     ← первая тестовая версия арены (референс)
└── specs/
```

## Наследование полей (правило воркспейса)

Единый источник — `[workspace.package]` в корне:

| Поле | Значение | В крейте |
|---|---|---|
| `version` | 0.1.0 — **одна на весь workspace**, бампается вся разом | `version.workspace = true` |
| `edition` | 2024 | `edition.workspace = true` |
| `authors` / `license` / `repository` / `homepage` | MIT, github.com/m62624/plugmem | `<поле>.workspace = true` |

Локальные поля крейта: `name`, `description`, `publish = false` (wasm,
testgen), `[lib] crate-type` (wasm), `[[bin]]` (cli, mcp).

## Зависимости — только через `[workspace.dependencies]`

Версии закреплены в корне (числа — из `specs/08`); крейты подключают
`имя = { workspace = true }` и могут только **сужать** (features), но не
задавать версию. Локальные крейты тоже объявлены там с `path` + `version` —
поэтому публикация на crates.io не потребует правок манифестов.

Проверенное правило подключения ядра: `plugmem-arena`/`plugmem-core` в
workspace-записи имеют `default-features = false`; std включает потребитель
(`features = ["std"]` у host/cli/mcp/wasm). Это гарантирует, что gate-сборка
`--no-default-features --target wasm32v1-none` всегда отражает реальный
no_std-срез.

## Конвенции features

| Feature | Где | Смысл |
|---|---|---|
| `std` (default) | arena, core | std-удобства; выключение = чистый `no_std + alloc` |
| `counters` | arena, core (пробрасывается) | детерминированные счётчики работы; zero-cost при выключении |

Пробрасывание: `plugmem-core/std = ["plugmem-arena/std"]` — включение фичи у
верхнего крейта включает её по всей цепочке.

## Прочее

- `plugmem-napi`: `crate-type = ["cdylib", "rlib"]` (rlib — чтобы нативный
  `cargo test` гонял логику обвязки, напр. skill-гейт), собирается через
  `@napi-rs/cli` в `.node`-аддон; npm — мета-пакет `plugmem` + platform-пакеты.
- `[profile.dist]`, cargo-dist-метаданные и CI появляются на этапе 5 —
  осознанно не сейчас.
- Новый крейт в workspace = запись в `members` + наследование всех полей +
  зависимости только через workspace-таблицу; для no_std-крейтов — проба из
  `specs/08` до первого коммита.

## Качество: линты и покрытие (действуют с каркаса)

- **Недокументированных публичных элементов не существует**: в корне
  `[workspace.lints.rust] missing_docs = "deny"` (+ `unsafe_op_in_unsafe_fn`,
  `rustdoc::broken_intra_doc_links`); каждый крейт подключён через
  `[lints] workspace = true`. Забыл док-коммент — сборка падает.
- **Язык: код и вся документация — только English.** Спеки ведутся на русском
  до завершения проектирования и переводятся на английский в самом конце
  (решение зафиксировано в `00`).
- **tarpaulin едет рядом с разработкой**: конфиг `tarpaulin.toml` в корне
  (паттерн elenchus) — метрика сужена до библиотечных крейтов
  (arena/core/host/testgen), бинарники и wasm-мост исключены с обоснованием
  в комментариях конфига, `reference/` не измеряется. Отчёты — в `coverage/`
  (gitignored). Локальный прогон — просто `cargo tarpaulin`; мандат цифр —
  в `specs/07` (100% arena/core, ≥90% остальные), CI-гейт — на этапе 5.

## Команды проверки (выполнены при создании каркаса)

```sh
cargo check --workspace                                             # весь воркспейс
cargo build -p plugmem-arena -p plugmem-core \
    --target wasm32v1-none --no-default-features                    # no_std-гейт
```
