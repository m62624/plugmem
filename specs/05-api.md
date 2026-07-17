# 05 — публичный API ядра

> `plugmem-core`: типы, глаголы, Config, ошибки, контракт Embedder-слоя.
> API синхронный (однопоточное ядро), `no_std + alloc`. Все `now` — u64
> unix-миллисекунды от хоста.

## Точка входа

```rust
pub struct Memory { /* арены, индексы, конфиг, скретчи */ }

impl Memory {
    /// Новая пустая база.
    pub fn new(cfg: Config) -> Result<Self, Error>;
    /// Подъём: снапшот + реплей журнала из Storage. None-снапшот => new(cfg).
    pub fn open<S: Storage>(store: &mut S, cfg: Config) -> Result<(Self, OpenReport), Error>;
    /// Подъём из готовых байтов (wasm-путь: хост уже принёс blob + журнал).
    pub fn from_bytes(snapshot: Option<&[u8]>, journal: &[u8], cfg: Config)
        -> Result<(Self, OpenReport), Error>;

    pub fn remember<S: Storage>(&mut self, s: &mut S, input: RememberInput<'_>)
        -> Result<RememberOutcome, Error>;
    /// Импорт/массовая запись: журналируется как последовательность Remember;
    /// similar-детекция отключаема флагом (skip_similar) для скорости импорта.
    pub fn remember_batch<S: Storage>(&mut self, s: &mut S,
        inputs: &[RememberInput<'_>], skip_similar: bool)
        -> Result<Vec<RememberOutcome>, Error>;
    pub fn recall(&mut self, q: RecallQuery<'_>) -> Result<RecallResult, Error>;
    pub fn revise<S: Storage>(&mut self, s: &mut S, target: FactId, input: RememberInput<'_>)
        -> Result<RememberOutcome, Error>;
    pub fn forget<S: Storage>(&mut self, s: &mut S, now: u64, id: FactId) -> Result<bool, Error>;
    pub fn link<S: Storage>(&mut self, s: &mut S, input: LinkInput<'_>) -> Result<(), Error>;

    /// Вся обслуживающая работа. Явно, никаких фонов.
    pub fn maintain<S: Storage>(&mut self, s: &mut S, now: u64) -> Result<MaintainReport, Error>;
    /// Полный образ + очистка журнала.
    pub fn snapshot<S: Storage>(&mut self, s: &mut S, now: u64) -> Result<(), Error>;
    /// Образ в байты (wasm-путь; журнал чистит хост).
    pub fn snapshot_bytes(&self, now: u64) -> Vec<u8>;

    pub fn stats(&self) -> Stats;
    pub fn get(&self, id: FactId) -> Option<FactView<'_>>;
    pub fn entity(&self, name: &str) -> Option<EntityId>;
}
```

`recall` принимает `&mut self` только ради скретч-буферов (историческая
альтернатива — interior mutability; отвергнута: явность дешевле). Данные recall
не меняет.

## Входы/выходы

```rust
pub struct RememberInput<'a> {
    pub now: u64,
    pub text: &'a str,                       // ≤ cfg.max_text
    pub entity: Option<&'a str>,             // субъект; создаётся лениво
    pub tags: &'a [&'a str],                 // ≤ 32
    pub links: &'a [(&'a str, &'a str)],     // (rel, target_entity), ≤ 16; рёбра entity→target
    pub vector: Option<&'a [f32]>,           // len == cfg.dim; квантуется внутри
    pub valid_from: Option<u64>,             // default now
}

pub struct RememberOutcome {
    pub id: FactId,
    pub entity: Option<EntityId>,
    /// Подсказки агенту: похожие/потенциально конфликтующие живые факты.
    /// Движок НИКОГДА не ревизит сам — решение за агентом.
    pub similar: Vec<Similar>,               // ≤ 8, по убыванию score
}
pub struct Similar {
    pub id: FactId,
    pub score: f32,
    pub reason: SimilarReason,               // SameEntity | LexicalOverlap | VectorClose
}
```

Детекция similar (дёшево, из уже готовых индексов): живые факты той же
сущности (∩ пересечение термов > 0.5 по Жаккару на топ-термах) ∪ векторные
соседи с cos > 0.85 (порог в Config). Это ключ к Graphiti-классу поведения
без LLM внутри: движок находит, агент решает (`revise` / оставить оба /
`forget`). Полная детекция (включая векторную) входит в бюджет remember
≤ 500 мкс (решено; см. `07`) — на фоне вызова эмбеддера движок всё равно
невидим.

```rust
pub struct RecallQuery<'a> {
    pub now: u64,
    pub text: Option<&'a str>,
    pub vector: Option<&'a [f32]>,
    pub tags: &'a [&'a str],
    pub entities: &'a [&'a str],
    pub as_of: Option<u64>,                  // default now
    pub range: Option<(u64, u64)>,           // окно recorded_at (эпизодика)
    pub k: usize,                            // default 8, ≤ 64
    pub token_budget: Option<usize>,         // default 512
    pub include_closed: bool,
    pub ef: Option<usize>,                   // HNSW ef_search override
}

pub struct RecallResult {
    pub facts: Vec<RecalledFact>,            // id, score, sources bitmask, text-ref,
                                             // recorded_at, valid interval, entity, tags
    pub edges: Vec<RecalledEdge>,            // рёбра, пройденные графовым источником
    pub rendered: String,                    // готовый блок для промпта
    pub truncated: bool,                     // упёрлись в бюджет
}
```

### Формат `rendered` (контракт, фиксируется тестами)

```
## memory
- [f42] user: предпочитает tokio (2025-11; активен) #pref
- [f17] user: жил в Москве (2023-01 → 2025-06; закрыт: f58) #location
- links: user —works_on→ plugmem
```

Однострочные записи, стабильный порядок (по score), ISO-даты по месяцу,
маркер закрытых интервалов, id для последующих revise/forget агентом. Пустой
результат → пустая строка (не «ничего не найдено» — не тратим токены).

## Config (полный, с дефолтами)

| Поле | Default | Прим. |
|---|---|---|
| `dim` | 0 | 0 = векторный слой выключен |
| `max_bytes` | 2 ГиБ | суммарный потолок пулов |
| `max_text` | 4096 | байт |
| `max_blob` | 64 КиБ | |
| `shards_facts / entities / edges / temporal / postings` | 1024 / 256 / 512 / 512 / 2048 | степени 2 |
| `bm25_k1 / b` | 1.2 / 0.75 | |
| `rrf_k` | 60 | |
| `w_bm25 / w_vec / w_graph / w_time` | 1.0 | веса RRF |
| `w_recency / half_life_days` | 0.25 / 180 | |
| `graph_depth / graph_decay` | 2 / 0.5 | |
| `similar_cos / similar_jaccard` | 0.85 / 0.5 | |
| `hnsw_m / m0 / ef_construction / ef_search` | 16 / 32 / 200 / 64 | |
| `flat_to_hnsw` | 24_000 | порог, уточняется бенчем |
| `fast_load` | false | пропуск xxh3 секций |

Config сохраняется в снапшоте; при `open` заданный конфиг сверяется:
несовместимые поля (dim, shards) при непустой базе → `Error::ConfigMismatch`
(смена dim = реиндексация, отдельная утилита в CLI v2).

## Ошибки

```rust
pub enum Error {
    CapacityExceeded { what: &'static str },
    TooLarge { what: &'static str, len: usize, max: usize },
    DimMismatch { got: usize, want: usize },
    NotFound(FactId),
    AlreadyClosed(FactId),          // revise поверх closed
    ConfigMismatch(&'static str),
    Corrupt(&'static str),          // снапшот/журнал
    UnsupportedVersion(u16),
    Storage(...),                   // обёртка ошибки Storage
}
```

Паника в ядре = баг по определению (фиксируется fuzz'ом и review-политикой).

## Семантика глаголов — сводка

| Глагол | Журнал | Эффект |
|---|---|---|
| remember | да | новый открытый факт + индексы + similar-подсказки |
| revise | да | закрыть target (valid_to = new.valid_from), новый факт с revises=target |
| forget | да | tombstone немедленно (recall не видит), физика — в maintain |
| link | да | ребро (upsert по (src,rel,dst)) |
| recall | нет | чистый запрос |
| maintain | да (маркер) | purge tombstones, компакция BlobHeap/ChunkPool, вливание vector-хвоста в HNSW / пересборки, пересчёт статистик |
| snapshot | сброс | полный образ + clear_journal |

`maintain` — единственное место, где стоимость O(база); все остальные глаголы —
микросекундные (бюджеты в `07`). Обёртки зовут его по своим политикам
(CLI — команда + авто после N операций; MCP — в idle; wasm — решает хост).

## Слой Embedder (`plugmem-host`, std)

```rust
pub trait Embedder {
    fn dim(&self) -> usize;
    /// Батч — обязателен в сигнатуре: провайдеры сильно дешевле батчами.
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
}
```

Реализации v1: `OllamaEmbedder` (HTTP `/api/embed`), `OpenAiCompatEmbedder`
(`/v1/embeddings`, любой совместимый), `NullEmbedder` (dim 0). HTTP-клиент —
`ureq` (блокирующий, маленький: ядро синхронно, async в цепочке не нужен).

**Встроенный локальный эмбеддер — v1.1 (решено).** Ориентир: feature-флаг
`local-embed`, модель — квантованный `multilingual-e5-small`, backend —
**CPU или GPU по выбору пользователя** (кейс: VRAM занят LLM — эмбеддинги
считает CPU). Кандидат-рантайм — candle (чистый Rust); финальный выбор — при
реализации v1.1, ядро это не затрагивает (Embedder-контракт уже готов).

Правило связки: обёртка сама вызывает `embed` перед `remember`/`recall`
(если эмбеддер сконфигурирован) и кладёт вектор в input. Ядро про Embedder
не знает. В wasm то же самое делает JS-объвязка через callback хоста
(см. `06`).

## План тестов

- Контрактные тесты каждого глагола (таблицы из `02` + сценарии similar).
- `rendered` — golden-тесты (формат = контракт).
- Ошибки: каждый вариант Error достижим тестом.
- Zero-alloc recall: счётчик-аллокатор в тест-харнессе, 0 аллокаций на
  эталонном recall после прогрева (см. `07`).
- Embedder-реализации: против локального мок-HTTP (не сеть в CI).

## Открытые вопросы

- ~~Пакетный `remember_batch`~~ — решено: в v1, сигнатура выше.
- `recall`-объяснимость — решено для v1: `RecalledFact.sources` (bitmask
  источников) + топ-3 совпавших терма при лексическом матче; расширенный
  `why` — по нуждам SKILL.md после обкатки.
