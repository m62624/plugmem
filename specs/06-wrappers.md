# 06 — обёртки: CLI, MCP, WASM/npm, skill, доставка

> Все обёртки — тонкие: разбор входа → ядро → рендер выхода. Логика памяти в
> обёртках запрещена (ревью-правило). Одинаковые возможности на всех
> поверхностях; различия только в транспорте и способе получения байтов.

## Общее: конфиг и обнаружение базы

Приоритет: флаг/параметр > env > конфиг-файл > default.

- База: `--db PATH` | `PLUGMEM_DB` | `./plugmem.db` если существует |
  `$XDG_DATA_HOME/plugmem/default.plugmem` (создаётся).
- Конфиг-файл: `$XDG_CONFIG_HOME/plugmem/config.toml` — секции `[engine]`
  (поля Config из `05`), `[embedder]` (`kind = "ollama" | "openai" | "none"`,
  `url`, `model`, `api_key_env`), `[maintenance]` (`auto_after_ops = 1000`,
  `journal_snapshot_bytes = 4МиБ`).
- Эмбеддер по умолчанию: `none` (система обязана работать из коробки без
  сервисов); включение — одна секция конфига или `PLUGMEM_EMBEDDER=ollama`.
- Блокировка: FileStorage держит эксклюзивный flock (см. `03`) — одна база =
  один процесс. CLI при базе, занятой (например) MCP-сервером, печатает
  «database is locked by another process (pid …)» и выходит с кодом 1.

## CLI (`plugmem`, крейт plugmem-cli)

Команды (все поддерживают `--json` для машинного вывода; человеческий вывод —
default):

| Команда | Суть |
|---|---|
| `plugmem remember "text" [--entity E] [--tag T]... [--link REL:ENTITY]... [--valid-from TS]` | remember; печатает id + similar-подсказки |
| `plugmem recall [QUERY] [--tag]... [--entity]... [--as-of TS] [--range A B] [-k N] [--budget N] [--closed]` | recall; человеческий вывод = `rendered` |
| `plugmem revise ID "text" [...]` | revise |
| `plugmem forget ID` | forget |
| `plugmem link SRC REL DST` | link |
| `plugmem show ID` | полная карточка факта (цепочка ревизий, рёбра) |
| `plugmem stats` | размеры, счётчики, конфиг |
| `plugmem maintain` | явный maintain + snapshot |
| `plugmem export --format jsonl` / `import` | дамп/загрузка фактов (переносимость, бэкап человекочитаемым) |
| `plugmem migrate` | миграция формата (feature migrate) |

Exit-коды: 0 ok; 1 ошибка входа/не найдено; 2 повреждение базы. `now` берётся
из системных часов **здесь** (единственное место времени). Время в аргументах —
ISO-8601 или unix-ms.

Cold start — критичный путь CLI (процесс на команду): бюджет из `07`
(load = чтение файла). Никаких лишних инициализаций до разбора команды.

## MCP (`plugmem-mcp`)

stdio JSON-RPC. Инструменты (схемы зеркалят CLI/ядро):

- `plugmem_remember { text, entity?, tags?, links?, valid_from? }` →
  `{ id, similar[] }` — в описании инструмента прямо сказано: «если similar
  содержит противоречие — реши: plugmem_revise или оставь оба»;
- `plugmem_recall { query?, tags?, entities?, as_of?, range?, k?, budget?, include_closed? }`
  → `{ rendered, facts[], edges[] }`;
- `plugmem_revise`, `plugmem_forget`, `plugmem_link`, `plugmem_show`,
  `plugmem_stats`;
- `plugmem_version`, `plugmem_about` — версия движка + указание загрузить
  version-matched skill.

Сервер владеет одной базой (путь: аргумент/env как в CLI), эмбеддер — из того
же конфига. `maintain` — автоматически по политике `[maintenance]` в моменты
между запросами (не фоновый поток: проверка после каждого вызова).
SKILL.md встроен `include_str!` и отдаётся инструментом `plugmem_skill`.

## WASM / npm (`plugmem-wasm`)

Крейт: `crate-type = ["cdylib", "rlib"]`, publish = false (идёт в npm),
wasm-opt `-Oz`, target `wasm32-unknown-unknown` + проверка ядра под
`wasm32v1-none`. Сборка npm-пакета — скрипт `scripts/build-npm.mjs`
(паттерн elenchus): wasm-pack + ручная обвязка `npm/index.js` + `index.d.ts`.

JS-контракт (все callbacks синхронные — ядро синхронно; async-обвязка поверх —
дело хоста):

```ts
interface StorageHooks {
  readSnapshot(): Uint8Array | null;
  writeSnapshot(bytes: Uint8Array): void;
  readJournal(): Uint8Array;          // пусто = new Uint8Array(0)
  appendJournal(entry: Uint8Array): void;
  clearJournal(): void;
}
type EmbedHook = (texts: string[]) => Float32Array[];  // опционален

class Plugmem {
  static open(storage: StorageHooks, config?: EngineConfig, embed?: EmbedHook): Plugmem;
  remember(input: RememberInput): RememberOutcome;   // now обязателен в input
  recall(query: RecallQuery): RecallResult;
  revise(id: number, input: RememberInput): RememberOutcome;
  forget(now: number, id: number): boolean;
  link(input: LinkInput): void;
  maintain(now: number): MaintainReport;
  snapshot(now: number): void;
  stats(): Stats;
}
```

- `now` передаётся из JS (`Date.now()`) — симметрия с ядром: рантайм wasm
  часов не имеет.
- Node-удобство из коробки: `openFile(path, config?)` — готовые StorageHooks
  поверх `fs` (+ атомарная запись tmp+rename), `ollamaEmbedder(url, model)` —
  готовый EmbedHook через fetch (sync-мост: в Node — через
  `child_process.execFileSync`? **Нет** — решение: embed-хук в JS-обвязке
  async, обвязка вызывает его ДО синхронного вызова wasm-метода и передаёт
  готовый вектор; сам wasm-API принимает `vector?: Float32Array` в input —
  зеркально ядру. EmbedHook — сахар уровня JS-класса, не wasm-моста).
- TS-типы — полные, вручную написанные (не сгенерённые), с doc-комментариями.
- Тест-набор: `node --test` smoke + паритет с нативом (тот же сценарий →
  тот же rendered).

## SKILL.md (репо-корень `skill/`, встраивается в MCP и wasm)

Контент (version-matched к движку, паттерн elenchus):

1. Когда вспоминать: recall в начале задачи / при упоминании прошлого;
   пустой rendered — просто продолжай.
2. Когда запоминать: устойчивые факты, предпочтения, решения, идентичность —
   не эфемерщину; гранулярность «один факт = одно утверждение»; entity и
   теги — конвенции именования.
3. Цикл противоречий: remember → смотри similar → противоречие? →
   revise (изменилось) | оставить оба (совместимы) | forget (ошибка было).
4. Темпоральность: valid_from для «с прошлого месяца», as_of для «а что было
   тогда», range для эпизодики.
5. Worked examples на каждую поверхность (CLI-команды и MCP-вызовы).

## Доставка

Зеркало релизного конвейера elenchus: cargo-dist (installers shell/powershell/
msi/homebrew, cargo binstall поверх dist-manifest) для `plugmem` и
`plugmem-mcp`; библиотеки на crates.io (`dist = false`); npm-пакет — из CI по
тегу (OIDC). SKILL.md — артефакт каждого релиза, version-pinned. Матрица
таргетов: linux/windows/macos × x64/arm64.

## План тестов

- CLI: интеграционные через `assert_cmd`-стиль (бинарь + временная база):
  каждый субкоманда happy-path + ошибки; `--json`-схемы фиксированы golden'ами.
- MCP: сценарные JSON-RPC сессии (стенд как elenchus-mcp): tools/list,
  каждый инструмент, битые входы → корректные JSON-RPC ошибки.
- WASM: smoke в Node (`node --test`): open→remember→recall→snapshot→reopen;
  паритет rendered с нативом на общем сценарии (shared testdata).
- Кросс-поверхностный паритет: один YAML-сценарий операций, прогоняемый
  тремя раннерами (cli/mcp/wasm) — идентичные структурные результаты.

## Открытые вопросы

- `plugmem-server` (HTTP, много клиентов) — вне v1; когда появится, это
  отдельный крейт поверх core с настоящей конкурентной обвязкой
  (single-writer + snapshot-читатели).
- Импорт «сырых» диалогов с автоэкстракцией (extraction-модуль с LLM-вызовами) —
  осознанно вне ядра; возможный отдельный крейт v2.
