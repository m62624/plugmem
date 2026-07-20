# 15 — CI/CD (зеркало elenchus, улучшенное) и план оставшихся работ

> **Статус: CI/CD реализован 2026-07-20** (Fable). Конвейер скопирован с
> эталонного воркспейса elenchus (файлы `.github/` читались напрямую) и
> усилен под наши сложности: фиче-матрица clippy/тестов, исполнение
> сьюта ядра в двух wasm-рантаймах, кросс-таргет проба wasm32/wasm64.
> Вторая половина документа — handoff: что осталось по проекту и как
> продолжать (написано так, чтобы Opus мог подхватить без контекста).

## 1. Состав `.github/` и отличия от elenchus

| Файл | Откуда | Что делает / что изменено |
|---|---|---|
| `workflows/ci.yml` | elenchus, **усилен** | см. §2 |
| `workflows/bin-release.yml` | elenchus (генерат cargo-dist 0.31.0 + их правки) | сборка бинарей по матрице dist; **WiX v3 ставится руками** (candle.exe пропал с раннеров GitHub); правки: homebrew-репо → `m62624/homebrew-plugmem`, описания формул → plugmem-cli/plugmem-mcp |
| `workflows/release.yml` | elenchus | оркестратор `pin/v*` (§4); правки: skill-гейты самовзводящиеся, сентинел `<!-- plugmem-changelog -->` |
| `workflows/mirror-tangled.yml` | elenchus, 1:1 | зеркало main+тегов на Tangled по push в main |
| `workflows/labeler.yml` | elenchus, 1:1 | лейблы PR по conventional-префиксу заголовка (+breaking по `!`) |
| `workflows/coverage.yml` | elenchus, 1:1 | tarpaulin → Codecov; **не** в required-гейте (информационный) |
| `workflows/bench.yml` | elenchus | criterion по лейблу `benchmark`; правка: `-p plugmem-arena` + `-p plugmem-core` |
| `release.yml` (категории) | elenchus, 1:1 | категории авточенджлога по лейблам |
| `dependabot.yml` | elenchus | weekly, минор+патч одним групповым PR, мажоры cargo — руками |
| `codecov.yml` (корень) | elenchus | оба статуса informational; ignore зеркалит tarpaulin.toml |

Закреплённые версии инструментов (бампать **рукой**, с локальной
ревалидацией — Dependabot их не видит):

- **cargo-dist 0.31.0** — в трёх местах в ногу: `Cargo.toml`
  `[workspace.metadata.dist] cargo-dist-version`, `ci.yml` job `dist-plan`
  (URL инсталлера), `bin-release.yml` (URL инсталлера).
- **wasmtime v41.0.3 / wasmer v7.2.0** — `env` в `ci.yml`; это версии, на
  которых валидирована кросс-таргет эквивалентность (specs/14 §3).

## 2. `ci.yml` — джобы (все параллельны, гейт один)

1. **check** (матрица Linux/Windows/macOS, fail-fast off): fmt →
   **clippy ×4 комбо** (`default` / `no-default` / `counters` /
   `no-default+counters`, `-D warnings`) → **тесты ×2**
   (`default` и `--features counters` — перф-гейты живут только под
   counters). Фичи меняют компилируемый код — одна комбинация не
   покрывает остальные (specs/07 §8).
2. **no_std**: сборка arena+core под `wasm32v1-none`, обе комбинации
   counters.
3. **wasm-suite** (матрица {wasmtime, wasmer} × {default, counters}):
   **полный контрактный сьют ядра на настоящем 32-битном таргете**
   `wasm32-wasip1`, release. proptest-секции нативные (погейчены в самих
   тестах). Именно этот прогон поймал 32-битный abort в журнале
   (specs/14 §5).
4. **wasm-equivalence**: `tools/wasm-probe` собирается нативно (эталонный
   хеш), под `wasm32-unknown-unknown` (stable) и под
   `wasm64-unknown-unknown` (nightly, `-Zbuild-std=core,alloc`);
   артефакты исполняются: wasmtime (32 и 64), wasmer (32) — все обязаны
   напечатать эталонный хеш (снапшот байт-в-байт независим от
   разрядности). Отдельный **информационный** шаг пробует wasmer+wasm64:
   когда wasmer научится memory64, шаг сам подскажет обновить specs/14 и
   повысить его до обязательного.
5. **skill-lint** («проверка скилла в начале», на каждом PR): `skill/SKILL.md`
   существует, несёт корректный маркер `<!-- skill-version -->` и оба маркера
   `<!-- wasm-strip:begin/end -->`, frontmatter в лимитах Agent Skills
   (name ≤ 64, description ≤ 1024). **Равенство** маркера с версией
   проверяется дважды в других местах: юнит-тестом plugmem-wasm на каждом
   `cargo test` (скилл встроен `include_str!`) и релизным гейтом
   `skill-check` против версии, которую режет релиз.
6. **wasm-npm** (боевой, 2026-07-20): wasm-pack сборка + сборка пакета
   `scripts/build-npm.mjs` + smoke `node --test` — ровно то, что публикует
   релиз.
7. **dist-plan**: `dist plan` на закреплённой версии.
8. **ci-pass** — агрегатор. В branch protection required-чеком ставится
   **только он** («CI passed»): состав джобов меняется без правки
   настроек репо, и чек репортится на каждый PR (у PR нет paths-фильтра
   сознательно — required-чек не должен «висеть» на docs-only PR).

## 3. cargo-dist (бинарники)

- Конфиг: `[workspace.metadata.dist]` в корневом Cargo.toml. Бинарники —
  только `plugmem-cli` (bin `plugmem`) и `plugmem-mcp`; все библиотеки и
  tools несут `[package.metadata.dist] dist = false`.
- Таргеты: linux/windows/macos × x86_64/aarch64 (6).
- Инсталлеры: shell, powershell, **msi**, homebrew (+ `cargo binstall`
  бесплатно поверх dist-manifest.json).
- **Windows-фиксы** (перенос из elenchus): (а) `.msi` требует WiX v3 —
  step в bin-release.yml скачивает wix314-binaries и кладёт candle.exe в
  PATH; (б) `aarch64-pc-windows-msvc` собирается на **нативном**
  windows-2022 раннере (`[workspace.metadata.dist.github-custom-runners]`),
  потому что candle.exe не исполняется под cargo-xwin на Linux.
- WiX GUID'ы (`upgrade-guid`/`path-guid` в cli/mcp + `wix/main.wxs`)
  сгенерированы локальным `dist init` 2026-07-20 и **не меняются
  никогда** — это идентичность продукта для upgrade/uninstall в Windows.

## 4. Механика релиза (оркестратор, 1:1 elenchus)

```
git tag pin/v0.1.0 && git push origin pin/v0.1.0
```

`prepare` (bump версии workspace → ветка rc/v0.1.0, триггер-тег
удаляется) → `tests` (**этот же ci.yml** через workflow_call на RC) +
`skill-check` (маркер `<!-- skill-version -->` в skill/SKILL.md ==
режущаяся версия; **бампает человек** — это чекпоинт перечитать скилл) →
`tag` (настоящий v0.1.0) → `dist` (bin-release.yml, draft-релиз с
бинарями) → `release-notes` (наш ченджлог по лейблам PR поверх тела
dist, идемпотентно) + `skill-asset` (SKILL.md ассетом к релизу) →
`publish-crates`
(`cargo publish --workspace --locked`; publish=false-крейты — wasm,
testgen, tools — пропускаются сами; **порядок и ожидание индекса cargo
делает сам**) + `publish-npm` (OIDC trusted publishing, БЕЗ npm-токена) →
`sync` (PR rc→main с бампом версии).

Правило: crates.io и npm публикуются **только после** успешных
бинарей — неудачный релиз не сжигает неизменяемые версии.

## 5. Секреты и переменные (заполняет владелец репо)

Settings → Secrets and variables → Actions:

| Имя | Тип | Для чего | Как получить |
|---|---|---|---|
| `TANGLED_SSH_KEY` | secret | зеркало на Tangled | приватный ssh-ключ, публичная половина — в аккаунт Tangled |
| `TANGLED_REMOTE` | **variable** | адрес зеркала | вида `git@tangled.org:<handle>/plugmem` |
| `HOMEBREW_TAP_TOKEN` | secret | пуш формул в tap | PAT с правом push в репо `m62624/homebrew-plugmem` — **репозиторий надо создать** (пустой, с веткой main и папкой Formula/) |
| `CARGO_REGISTRY_TOKEN` | secret | crates.io publish | crates.io → Account Settings → API Tokens (scope publish-new + publish-update) |
| `CODECOV_TOKEN` | secret | загрузка покрытия | codecov.io после подключения репо |
| npm | — | publish-npm | токен НЕ нужен (OIDC), но **первая публикация** нового имени пакета делается вручную локально с npm-токеном, затем на странице пакета настраивается Trusted Publisher (repo+workflow release.yml) |

Ручная настройка репо: branch protection на `main` с единственным
required-чеком **«CI passed»**; лейбл `benchmark` (создастся сам при
первом использовании labeler'ом можно и руками).

## 6. Что осталось по проекту (handoff для продолжателя)

Сделано и закрыто: ядро (этапы 1–4 + HNSW этап 6), plugmem-host,
testgen, README core/host, wasm-классы ёмкости (specs/14), CI/CD (этот
док). Дальше — **этап 5** (specs/06 — источник истины по поведению):

1. **plugmem-cli** (`crates/plugmem-cli`, стаб готов): команды из
   таблицы specs/06 поверх `plugmem_host::Database`; exit-коды 0/1/2;
   `--json` (golden-тесты схем, insta уже в dev-deps); `now` — только из
   системных часов CLI; конфиг: флаг > env > config.toml > default.
   Тесты — интеграционные, спавнящие бинарь (main.rs исключён из
   покрытия, логику держать в lib-модуле крейта и покрывать ≥90%).
2. **plugmem-mcp**: stdio JSON-RPC, инструменты из specs/06 (схемы
   зеркалят ядро), `plugmem_skill` отдаёт встроенный `include_str!`
   SKILL.md; авто-maintain по политике между запросами (без фоновых
   потоков). Сценарные тесты JSON-RPC сессий.
3. **plugmem-wasm — движковый контракт** (обвязка уже готова,
   2026-07-20): пакетирование (`scripts/build-npm.mjs`), Node-вход
   `npm/index.js` + `index.d.ts`, smoke-тест и публикация — боевые;
   сейчас пакет честно экспортирует только `version/about/skill/
   skillFull/skillVersion` (стаб задокументирован в README крейта).
   Осталось: класс `Plugmem` (Storage/Embedder через JS-callbacks,
   контракт specs/06) — добавляется в `src/lib.rs` + `npm/index.*`, тесты
   в `test/smoke.test.mjs`; конвейер не трогать. Учесть specs/14 §4:
   второй артефакт wasm64 + опция `open({ memory64 })` — можно v1.1.
4. **skill/SKILL.md — осмысленный текст** (шаблон уже стоит,
   2026-07-20): каркас, frontmatter и оба маркера финальны
   (`<!-- skill-version: X.Y.Z -->` — сверка с версией движка;
   `<!-- wasm-strip:begin/end -->` — блок «Run it», вырезаемый из
   npm-дистрибуции скилла и Rust-аксессором `skill()`). Секции,
   помеченные «(stub)», расписать по плану specs/06; маркеры и
   структуру не менять — на них стоят три гейта (skill-lint в CI,
   юнит-тест wasm-крейта, релизный skill-check).
5. **Перевод всех specs на английский** — финальный шаг перед
   публикацией (языковая политика specs/00).
6. Первый релиз: заморозка формата снапшота (specs/03), затем
   `git tag pin/v0.1.0`.

Незыблемые правила (нарушение = стоп ревью): код/rustdoc/коммиты —
только English (спеки русские до п.5); трейлер коммита — только
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; никаких
магических числовых оффсетов — только именованные const, выводимые друг
из друга; коммитить только зелёное состояние ворот (specs/11 §5 +
wasm-сьют: `CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=."
cargo test --release -p plugmem-core --target wasm32-wasip1`); каждая
новая зависимость ядра — через пробу specs/08; бенчи не сравнивают с
чужими БД; покрытие ≥90% (цель 100% arena/core), tarpaulin-ложь
аудируется руками, не списывается.
