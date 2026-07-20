# 11 — План этапа 4: векторный слой, maintain, testgen, README (handoff)

Этот документ — исполняемый план оставшейся работы по plugmem-core, написанный
так, чтобы реализацию можно было продолжить без контекста предыдущих сессий.
Все решения здесь **уже приняты** — не пересматривать их без крайней нужды,
а при пересмотре обновить этот файл и связанные спеки (02, 03, 04, 05, 07).

## 0-bis. Пересмотр приоритета (пользователь, 2026-07-19): ядро до 100%, обёртки потом

Embedded-ценность проекта — «Rust-библиотека: указал файл и поехали».
CLI/MCP/WASM — транспорт для не-Rust потребителей, откладываются до
полного ядра. Порядок до «ядро 100%»:

1. ~~`db_uuid` в формат снапшота~~ — **реализовано 2026-07-19**;
2. ~~этап C — `plugmem-testgen`~~ — **реализовано 2026-07-19**;
3. ~~этап 6 — HNSW~~ — **реализовано 2026-07-20** (specs/10);
4. ~~этап D — README core + прогон стенда~~ — **реализовано 2026-07-20**
   (README + assets/benchmarks.svg + фактические цифры в specs/07);
5. ~~`plugmem-host`~~ — **реализовано 2026-07-20** (specs/13: FileStorage
   + Database + OpenAiCompatEmbedder + README). Host — часть Rust-опыта,
   не обёртка.

**Ядро (и host) завершены.** Дальше — этап 5 (CLI, MCP, wasm, SKILL.md)
и перевод спек на английский перед публикацией. CI/CD и релизный
конвейер уже стоят (2026-07-20) — состав, секреты и подробный handoff
этапа 5 в `specs/15-ci-cd-and-next.md`; wasm-классы ёмкости — в
`specs/14-wasm-versions.md`.

## 0. Порядок работ и почему именно такой

1. **Этап A — векторный слой** (самая большая ценность; трогает формат
   снапшота и журнала, поэтому идёт первым — всё остальное строится поверх
   стабильного формата).
2. **Этап B — `maintain`** (purge/компакция; идёт после векторов, потому что
   компакция должна уметь компактить и VecPool).
3. **Этап C — `plugmem-testgen`** (детерминированный генератор корпусов; после
   A и B, чтобы генерировать и векторы, и maintain-нагрузку).
4. **Этап D — README core + прогон стенда** (нужны финальные цифры после A–B).

Каждый этап — отдельный коммит (или несколько), коммитить **только зелёное**
состояние (см. §5 «Ворота качества»).

## 1. Этап A — векторный слой (specs/04 §6, specs/02)

> **Статус: реализовано 2026-07-19** (Opus по этому плану). `index/vecpool.rs`,
> интеграция в verbs/recall/similar/journal/snapshot, тесты `tests/vectors.rs`
> + юнит-тесты в vecpool + counters-гейт; vecpool покрыт на 100%, ворота
> зелёные, wasm32v1-none собирается. Детали ушли в `specs/04 §5`. Осталось
> по плану: этапы B (maintain), C (testgen), D (README/стенд).

Цель: flat-поиск по квантованным векторам как четвёртый источник recall,
плюс векторный сигнал в similar-detection. Без HNSW (это этап 6, specs/10).

### A.1 Хранилище: `VecPool` (новый файл `crates/plugmem-core/src/index/vecpool.rs`)

Не использовать `BlobHeap` и не добавлять ничего в plugmem-arena: flat-поиск
требует идеальной локальности, поэтому VecPool — плоский `Vec<u8>` со
слотами фиксированного шага. Никакого переиспользования дыр — слоты только
аппендятся, мёртвые вычищает `maintain`.

```rust
pub(crate) struct VecPool {
    bytes: alloc::vec::Vec<u8>,
    dim: usize,      // из Config, > 0
    max_bytes: usize // cfg.max_bytes; превышение → Error::Arena(CapacityExceeded)-класс
}
```

Раскладка одного слота, все числа little-endian, шаг
`stride = 4 + 4 + 8 * words + dim`, где `words = dim.div_ceil(64)`:

| off | size | поле |
|---|---|---|
| 0 | 4 | `fact` — `FactId` владельца (u32 LE) |
| 4 | 4 | `scale` — f32 LE, масштаб квантования |
| 8 | 8·words | `sig` — битовая сигнатура, u64 LE на слово; бит `i` = `q[i] >= 0` |
| 8+8·words | dim | `q` — i8-компоненты |

Число слотов = `bytes.len() / stride`; отдельного счётчика в engine state
**не заводить** (STATE_LEN остаётся 24) — количество выводится из длины,
валидатор проверяет `len % stride == 0`.

Квантование (детерминированное, одна функция, используется и для записи,
и для запроса):

1. Вход `&[f32]`, длина строго `cfg.dim`, иначе `Error::Invalid`.
2. L2-нормировка: `norm = sqrt(Σ x²)` через `libm::sqrtf`; `norm == 0`
   или не-finite вход → `Error::Invalid("vector must be finite and nonzero")`.
3. `scale = max(|x_i|/norm) / 127.0`; `q_i = roundf((x_i/norm) / scale)`
   через `libm::roundf`, кламп в `[-127, 127]` (не −128 — симметрия).
4. Косинус двух слотов: `cos(a,b) ≈ scale_a · scale_b · Σ (qa_i as i32 · qb_i as i32)`.
   Аккумулировать в `i32` (dim ≤ 4096, |q| ≤ 127 → максимум 4096·127² < 2³¹ — влезает).

Поиск `search(&self, query_slot: &[u8], k, admit: &mut dyn FnMut(FactId) -> bool, scratch, out)` —
двухфазный:

1. **Фаза 1 (сигнатуры):** для каждого слота hamming = Σ `(sig_q[w] ^ sig_i[w]).count_ones()`.
   Собрать в scratch-вектор `(hamming u32, slot u32)`, `sort_unstable`,
   взять первые `C = max(4*k, 64)` кандидатов.
2. **Фаза 2 (точный dot):** для C кандидатов посчитать квантованный косинус,
   пропустить не прошедших `admit` (тот же admit-замыкание, что у BM25:
   tombstone/as_of/allow-set), отобрать top-k по убыванию скора в `out`
   (≤ SOURCE_CAP = 128, как у остальных источников).

Counters-гейт: при `feature = "counters"` считать точные dot'ы через
`Cell<u64>` (по образцу `Bm25Index::decoded`) и в perf-гейте утверждать
`dots == min(C, слотов)` — фильтр по сигнатурам обязан ограничивать работу.

### A.2 Интеграция в Memory

- `Memory::new`: **убрать** отказ `cfg.dim != 0` («vector layer lands in
  stage 4»); при `dim > 0` создавать `VecPool`. Обновить тест, который
  проверял этот отказ.
- `RememberInput` получает поле `vector: Option<&'a [f32]>` (в `text()`
  конструкторе — `None`). При `Some`: длина == dim (иначе `Error::Invalid`),
  при `dim == 0` любое `Some` → `Error::Invalid`.
- Путь записи (`apply_remember`): квантовать, аппендить слот в VecPool,
  в `FactRecord` ставить `vector = индекс слота` и флаг
  `fact_flags::HAS_VECTOR`. Без вектора — `vector = NONE_U32`, флага нет.
- `RecallQuery` получает `vector: Option<&'a [f32]>` (в `text()` — `None`).
- Источник recall: бит `VEC: u8 = 1 << 3` (BM25=1, GRAPH=2, TIME=4 заняты),
  вес `cfg.w_vec`, вливается в общий RRF как четвёртый источник. Scratch'и
  (вектор hamming-пар + буфер квантованного запроса размером stride) — поля
  `RecallScratch`, чтобы инвариант zero-alloc сохранился (тест
  `tests/zero_alloc.rs` обязан остаться зелёным; помнить про 2 прогревочных
  прохода).
- Similar-detection: если у нового факта и у кандидата есть векторы —
  считать квантованный косинус; `> cfg.similar_cos` → сигнал. В
  `SimilarReason` добавить вариант `VectorCosine`; при обоих сигналах
  берётся больший скор, `reason` — победивший. Кандидатский ring
  (SIMILAR_CANDIDATE_CAP=32) не менять.

### A.3 Журнал

Формат pre-release — ломаем без миграции (обновить specs/03). Payload
`Remember`/`Revise` расширяется хвостом: `[vec_dim u32 LE][f32 LE × vec_dim]`,
где `vec_dim ∈ {0, cfg.dim}` (иначе `Error::Corrupt`). В журнале лежит
**исходный f32 до квантования** — replay квантует заново той же функцией
(она детерминирована), состояние обязано сойтись байт-в-байт (это проверяет
существующий canonical-snapshot тест + новый replay-тест с векторами).
Проверку «op-revises agreement» и trailing-bytes-check в `journal.rs`
сохранить.

### A.4 Снапшот

- Новая секция `kind::VEC_POOL = 37` — сырые байты `VecPool.bytes`
  (одна секция; meta не нужна — dim и stride выводятся из Config).
  Каноничность: секции по возрастанию kind, как сейчас.
- `validate_references` (в `memory/persist.rs`) — **снять** ворота v1
  `vector == NONE_U32` и заменить на: `len(VEC_POOL) % stride == 0`;
  для каждого факта: если `HAS_VECTOR` — `vector < slot_count` и
  `слот.fact == id факта` (биекция), иначе `vector == NONE_U32`;
  `scale` каждого слота — finite и `>= 0`; каждый бит `sig` обязан
  совпадать с `q_i >= 0` (пересчёт дёшев и закрывает контрактные паники
  навсегда — принцип «после успешного load ни один сохранённый байт не
  может вызвать панику» обязателен).
- `kind == 0` в v1 остаётся обязательным.
- `dim` уже входит в структурные ворота Config (equality gate) — не трогать.
- При `dim == 0` секция VEC_POOL пишется пустой (0 байт) — грузится в
  пустой пул; отсутствие секции = `Corrupt` (все 37 секций обязательны).

### A.5 Тесты этапа A (файл `tests/vectors.rs` + правки существующих)

1. **Квантование, property:** 200 случайных пар unit-векторов (детерминированный
   PCG/xorshift с фиксированным сидом, dim 384): `|квант-косинус − f32-косинус| < 0.05`.
2. **Golden:** маленький ручной пример (dim 4, значения посчитаны независимо,
   в комментарии — процедура на Python, как у BM25-golden в `tests/index.rs`).
3. **Поиск:** корпус 1000 векторов, запросы: top-k квант-поиска против
   brute-force f32-косинуса — recall@8 ≥ 0.9 (сигнатурный фильтр не идеален,
   гейт с запасом; если стабильно выше — ужесточить по факту).
4. **Admit:** tombstone/as_of/закрытые факты не попадают в выдачу vec-источника.
5. **Персистентность:** roundtrip snapshot с векторами каноничен
   (save→load→save байт-в-байт), битфлип-свип по образцу
   `tests/persist.rs::corrupt_snapshots_are_typed_errors` — включая порчу
   внутри VEC_POOL (типизированная ошибка, не паника).
6. **Журнал:** replay c векторами эквивалентен прямому исполнению
   (property по образцу существующего replay-теста); `vec_dim` ≠ {0, dim}
   → `Corrupt`.
7. **Zero-alloc:** существующий тест дополнить запросом с вектором.
8. **Errors:** NaN/нулевой/неверной длины вектор → `Error::Invalid`;
   вектор при `dim == 0` → `Error::Invalid`.
9. **Бенч** в `benches/engine.rs`: группа `vec` — 24 000 векторов dim 384,
   поиск k=8; бюджет по specs/07 — worst < 1 ms нативно (замерить и вписать
   фактическое в specs/04 §6 и README).

## 2. Этап B — `maintain` (purge + компакция)

> **Статус: реализовано 2026-07-19** (Opus). `memory/maintain.rs`:
> пересборка сателлитов (texts/tag_lists/bm25/tags_idx/entity_facts/temporal/vecs
> + facts/entities/aux) при стабильных id; tombstone'ы остаются записями с
> зачищенным payload (пустой блоб, без вектора/тегов); интернер/by_name/рёбра
> не трогаются; ре-токенизация живых через lookup, копия квантованных слотов;
> детерминированный обход по id. `Op::Maintain` журналируется до swap, replay
> переисполняет ту же компакцию → снапшот байт-в-байт. `MaintainReport
> { purged, bytes_before, bytes_after }`. Тесты `tests/maintain.rs` (8, вкл.
> proptest observation-equivalence): сохранение состояния, reclaim,
> каноничность+replay, ноль орфанов (roundtrip), стабильность id/цепочек/рёбер,
> векторы, empty+идемпотентность. maintain.rs покрыт на 98% (остаток —
> tarpaulin false-negative на полях struct-литерала), ворота зелёные, wasm ок.
>
> **Дополнено (Fable, 2026-07-19): maintain v2** — tombstone-записи теперь
> удаляются физически (см. пересмотренный пункт B.1 и specs/12 §7-bis);
> добавлен `stats()`/`Stats` по specs/05.

Сейчас `Op::Maintain { now }` — no-op маркер. Становится реальной операцией.

### B.1 Решения (приняты; пункт о скорлупах пересмотрен 2026-07-19)

- **FactId/EntityId/TermId стабильны навсегда.** Никакого перенумерования:
  внешние ссылки, revises-цепочки и рёбра держатся на id.
- ~~Tombstone-факты не удаляются из арены фактов~~ **Пересмотрено
  (maintain v2, specs/12 §7-bis): tombstone-записи удаляются физически** —
  `FactRecord`/`FactAux` не переносятся в пересобранные арены. Id
  «сжигается»: аллокация идёт от персистентного `next_fact`, дырка в
  нумерации законна (specs/02), сожжённый id ведёт себя как tombstone
  (`get` → None, глаголы → NotFound). Ссылки на вычищенный факт
  (`revises`, provenance рёбер) сохраняют сожжённый номер — резолв даёт
  `None` в обоих мирах, что и обеспечивает наблюдаемую эквивалентность.
- Интернер **не** перестраивается (TermId-стабильность; мусорные термы —
  кандидат на v2, зафиксировать в specs/04 как известный компромисс).
- Рёбра графа сохраняются все (это самостоятельное знание; provenance-id
  остаётся валидным, т.к. слот факта не удаляется).
- `maintain` **не** делает снапшот сам — пишет `Op::Maintain { now }` в
  журнал; replay обязан исполнить maintain заново и сойтись байт-в-байт
  (алгоритм обязан быть детерминированным — см. порядок обхода ниже).

### B.2 Алгоритм (rebuild со стабильными id)

Сигнатура: `pub fn maintain<S: Storage>(&mut self, store: &mut S, now: u64) -> Result<MaintainReport, Error>`;
`MaintainReport { purged: usize, bytes_before: usize, bytes_after: usize }`.

Построить **новые** структуры рядом со старыми, затем свапнуть; порядок
обхода строго фиксирован (каноничность снапшота):

1. Новый `BlobHeap` текстов: сначала имена сущностей в порядке
   `EntityId 0..next_entity` (обновить `EntityRecord.name`), затем один
   пустой блоб `""` (общий для всех tombstone), затем тексты живых фактов
   в порядке `FactId 0..next_fact`.
2. Новые: tag `ChunkPool`, три `PostingStore`, doc_len-арена, temporal-арена,
   `VecPool`. Для каждого **живого** (не tombstone) факта в порядке id:
   скопировать текст, перечитать теги из старого aux → записать в новый пул,
   переиндексировать BM25 (`index_doc` — токенизатор детерминирован,
   постинги по возрастанию id сохраняются автоматически), tags_idx,
   entity_facts, temporal-вставка, копия векторного слота (с новым индексом
   в `FactRecord.vector`).
3. Для каждого tombstone: `text = пустой блоб`, `vector = NONE_U32`,
   флаг `HAS_VECTOR` снят, aux → `EMPTY`.
4. Пересчитать `total_docs`/`total_len` заново по живым.
5. Свап, `journal Op::Maintain { now }` (мутация до журнала недопустима при
   ошибке журнала — порядок «check first, mutate last»: собрать всё новое,
   записать в журнал, свапнуть; своп инфаллибелен).
6. В `replay` ветку `Maintain` заменить с no-op на исполнение того же
   алгоритма (без записи в журнал).

### B.3 Тесты этапа B (файл `tests/maintain.rs`)

1. **Эквивалентность recall:** случайный workload → снять выдачи recall по
   набору запросов → `maintain` → выдачи идентичны (включая rendered-байты).
2. **Reclaim:** запомнить крупные тексты, `forget` половину, `maintain` →
   `bytes_after < bytes_before`; снапшот после меньше снапшота до.
3. **Каноничность/replay:** журнал с Maintain внутри → replay → снапшот
   байт-в-байт равен снапшоту исходного движка.
4. **Орфаны:** после maintain `ChunkPool::orphan_count == 0` по всем пулам
   (валидатор загрузки это уже умеет — прогнать snapshot roundtrip).
5. **Property:** случайный workload со вставленными maintain в случайных
   точках эквивалентен по видимому состоянию (get/recall) тому же workload
   без maintain.
6. **Векторы:** maintain выбрасывает векторы tombstone'ов, живые ищутся
   как прежде; биекция `slot.fact ↔ record.vector` сохраняется.

## 3. Этап C — `plugmem-testgen`

> **Статус: реализовано 2026-07-19** (Fable). Крейт `plugmem-testgen`:
> собственный xorshift64* (без rand_xoshiro — зависимость снята из
> workspace), слоговые Zipf-словари (`word_for` — чистая функция индекса),
> `Profile`/`GenOp`/`Gen` + хелпер `apply` (единственное место маппинга на
> глаголы движка). Все операции валидны по построению (генератор ведёт
> собственную бухгалтерию open/live). Тесты: детерминизм (в т.ч. чанки ≡
> одному вызову), чистое применение 1200 ops с maintain'ами + канонический
> реплей журнала, unit-нормы векторов, Zipf-форма, монотонная уплотняющаяся
> ось времени. Бенчи core переведены на testgen-корпус; добавлена группа
> `vec` (A.5(9)): flat 24k × dim 384, k=8 → 332 мкс (< 1 мс бюджет).

Новый крейт `crates/plugmem-testgen` (обычный std-крейт, dev-инструмент,
**не** входит в wasm-паспорт). Содержимое:

- Детерминированный PRNG — PCG32 или xorshift64*, реализованный на месте
  (~15 строк, без новых зависимостей; специзменение в specs/08 не нужно).
- Zipf-сэмплер по словарю N слов (генерация слов — слоговая: согласная+гласная,
  чтобы токенизатор работал по-настоящему).
- Пул сущностей, пул тегов, генератор псевдослучайных unit-векторов.
- API: `Gen::new(seed: u64, profile: Profile) -> Gen`;
  `gen.ops(n) -> Vec<GenOp>` где `GenOp` зеркалит операции движка
  (Remember/Revise/Forget/Link/Maintain) с owned-строками и опциональным
  вектором; `Profile` задаёт доли операций, размер словаря, dim.
- Использование: заменить ad-hoc Zipf-корпус в `benches/engine.rs`,
  использовать в property-тестах maintain/replay.
- Опционально bin `testgen` (флаг `--jsonl`) — дамп потока операций для
  внешних стендов сравнения (specs/07 §5).

## 4. Этап D — README core + прогон стенда

1. `crates/plugmem-core/README.md` (на английском, как весь код): что это,
   архитектура в 10 строк (arena → indexes → memory verbs → snapshot),
   quickstart (remember/recall/snapshot/open), таблица замеров (взять из
   specs/07 фактические числа: recall @100k = 61 µs, BM25 65 µs, вставка
   ~1 µs/doc + новые цифры vec/maintain), команды воспроизведения бенчей,
   wasm-заметка (wasm32v1-none, без std).
2. Прогон стенда: полный feature-matrix тестов и clippy (§5), `cargo bench
   -p plugmem-core` нативно, `cargo build --target wasm32v1-none
   --no-default-features -p plugmem-core` — проверка паспорта, tarpaulin
   ≥ 90 % (цель 100 % по arena/core), свежие числа вписать в specs/07,
   README и, при расхождении, в specs/04.

## 5. Ворота качества (перед КАЖДЫМ коммитом)

```sh
cargo fmt --all
# clippy: 4 комбинации фич, 0 предупреждений
cargo clippy --workspace --all-targets                                  # default (std)
cargo clippy --workspace --all-targets --no-default-features            # no_std
cargo clippy --workspace --all-targets --features counters              # std+counters
cargo clippy --workspace --all-targets --no-default-features --features counters
cargo test --workspace
cargo test --workspace --features counters   # perf-гейты со счётчиками
cargo build --target wasm32v1-none --no-default-features -p plugmem-core -p plugmem-arena
```

Правила: rustdoc на всём публичном (`missing_docs = deny`), весь код и доки
— английский; спеки — русский до финального перевода. Трейлер коммита —
только `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, больше
никаких трейлеров.

## 6. Ловушки, уже пойманные в этом проекте (не наступать повторно)

- `hashbrown` без фичи default-hasher: для scratch-HashMap явно указывать
  `xxhash_rust::xxh3::Xxh3Builder` (и это даёт детерминизм).
- no_std: никакой `f32::ln/sqrt/round` — только `libm` (`logf`, `sqrtf`,
  `roundf`); constants из `core::f32::consts`.
- Zero-alloc тест — счётный глобальный аллокатор в отдельном бинаре;
  любые новые scratch'и recall кладутся в `RecallScratch` и прогреваются
  двумя проходами.
- Заимствования полей `RecallScratch` одновременно — деструктурировать:
  `let RecallScratch { allow, graph_out, .. } = s;`.
- Порядок мутаций в глаголах: «check first, mutate last» — сначала все
  проверки и сборка, журнал, затем инфаллибельные свопы/вставки.
- Каноничность: save→load→save обязан быть байт-в-байт; любой новый кусок
  состояния должен либо сериализоваться детерминированно, либо выводиться.
- Бенч-корпуса — только Zipf по реальному словарю (одинаковые 6 текстов
  дают df ≈ N и ломают смысл замера).
- Паника в ядре = баг: любые id из снапшота валидируются загрузчиком до
  adoption; новые ссылки (VEC_POOL) — не исключение.
- `cargo tarpaulin` гоняет proptest с другими сидами — падения под
  tarpaulin реальны, не списывать на инструмент.
