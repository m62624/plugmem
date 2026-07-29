# 04 — Indexes and hybrid recall

One fact store, several entry points into it: lexical (BM25), semantic (vectors),
structural (tags, entities, graph), temporal. Recall fuses them into one ranked
result and assembles a compact context under a token budget.

## 1. Tokenizer (in the core: needed on wasm too)

The tokenizer is the stack the top lexical engines converge on (Lucene ICU/Standard,
SQLite FTS5 `unicode61`), built from pure `core` unicode-rs crates (+163 KiB of
tables in the wasm binary, accepted):

1. **NFKC normalization** of the input (fullwidth → ASCII, ligatures ﬁ → fi,
   decomposed marks recombined). Pure ASCII takes an identity fast path (~35%).
2. **UAX #29 segmentation** (`unicode-segmentation`, the ICU standard). Consequences:
   `don't`/`o'clock` stay whole (an apostrophe joins letters); `3.14`, `v1.2.3`,
   `1,000`, `docs.rs` stay whole; `snake_case` stays whole; `gpt-4o` splits on the
   hyphen.
3. **Token folding**: full Unicode lowercase; Latin diacritics fold to the ASCII base
   (`café` → `cafe`, the FTS5 `remove_diacritics` class) **only** for Latin —
   Cyrillic `й` is left alone (the classic unicode61 bug); Russian `ё` → `е`. Tokens
   are canonical: tokenization is a fixed point (a property test).
4. **CJK**: Han and Hiragana come out of UAX #29 per character → adjacent ones are
   glued into overlapping **bigrams** (the Lucene `CJKBigramFilter` scheme, dictionary
   free); a lone character is a unigram. Katakana and Hangul segment into words and
   stay words.

No stemming/lemmatization in v1. A token → `TermId` via the interner; longer than
64 B is cut at a character boundary. The `Tokenizer` uses reusable scratch buffers
(zero allocations after warm-up — the recall invariant). Measured throughput: ASCII
~17 ns/B, mixed ru/en ~29 ns/B, CJK ~19 ns/B → a worst-case `max_text` of 4 KiB is
~70–120 µs (within the remember budget); a typical 150 B is ~3–5 µs.

## 2. Lexical index (BM25)

Structures:

- `postings: Arena<PostingHandle>` (Uniform, key TermId, 20 B slot: term 4 +
  ListHandle 12 + df u32) — term → document list.
- Postings in a `ChunkPool`: a sequence of `[delta(fact_id) varint][tf u8]`,
  fact_ids ascending (facts are inserted monotonically by id ⇒ append to the tail,
  sorting is free).
- `doc_len: Arena<DocLen>` (Uniform, 8 B slot) plus global `total_docs`, `total_len`.

Scoring is classic BM25 (k1 = 1.2, b = 0.75, in Config). A query's terms → decode
each posting accumulating `score[fact] += idf·tf_norm` in a reusable open-addressing
scratch → top-K by sorting a small buffer. Cost is O(Σ df); no WAND heuristics at our
scale (a v2 optimization at million-df).

A **stop-frequency filter**: a query term with df > corpus/8 (and df > 1024) is
dropped — under idf weighting it barely discriminates yet its posting list dominates
cost (a query with a "the"-class word cost O(corpus): 2.5 ms on 100k). If *all* query
terms are stop-frequent, the least frequent is kept (a query always answers). After
the filter, structural recall @100k is ~61 µs (budget 200 µs); the rule is
deterministic and covered by a bench test. Tombstone/closed facts are filtered per
candidate (a bit-check on the record) and fall out of the postings on the `maintain`
rebuild.

## 3. Tags and entities

- tag → facts: `Arena<TagHandle>` (key TermId) + ChunkPool lists of varint fact_id
  deltas (sorted by construction).
- A query's tag filter: intersection of the sorted lists (merge), giving an allow-set
  for the other sources (a bitmap in scratch).
- entity → facts: the same scheme (`EntityHandle`), filled on remember.

## 4. Time index

The arena from `02-data-model.md` (key `[recorded_at BE | fact_id]`). Query modes:
`range(a, b)` (facts recorded in a window — episodic memory) via a range scan;
`as_of(t)` as a validity filter on any source's candidates (O(1) per candidate); and
a recency boost in ranking (see fusion).

## 5. Vector layer

Config: `dim` (0 = layer off; ≤ 4096), cosine metric — vectors are normalized on
intake, then it is all dot product. Storage is flat sections:

- `vec_i8`: slot = `dim` × i8 + `scale f32` (symmetric quantization:
  `scale = max|x|/127`, `q = round(x/scale)`; ~1–2% recall loss, accepted).
- `sigs`: 1-bit-per-dim binary signatures packed into u64 words (48 B at 384d) — a
  pre-filter.
- f32 is stored nowhere (the capacity contract).

Quantization happens **before** the journal (the journal already holds i8+scale), so
replay is deterministic and needs no float reproducibility; a replay re-quantizes with
the same pure function and reproduces the slot byte-for-byte. The `VEC_POOL` snapshot
section's loader checks the fact↔slot bijection, a finite scale, and that signatures
match the signs (panic-free on any bytes).

The layer is a fourth recall source (bit `VEC`, weight `w_vec`) and a similar-detection
signal (`SimilarReason::VectorCosine`, threshold `similar_cos`).

### Search, phase 1 — Flat (below the threshold)

1. Hamming scan of the query signature over all live vectors (popcount over u64,
   auto-vectorizable, SIMD128-friendly): top `R = max(4·k, 256)` candidates.
2. Rescore R by i8-dot (i16→i32 accumulation) → exact top-k.

At @100k a full signature scan is ~5 MB → ~0.5 ms, the budget edge — hence a
threshold (~24k vectors, tuned by a bench).

### Search, phase 2 — HNSW (above the threshold)

- Parameters: M = 16, M0 = 32, standard level geometry; `ef_construction = 200`,
  `ef_search` default 64 (overridable per query via `RecallQuery.ef`).
- Storage: level 0 is fixed 32-neighbour × u32 slots (128 B) in a DynArena; upper
  levels are ChunkPool lists; entry point + counters in a header.
- In-graph distance is i8-dot over `vec_i8` (signatures are not used in HNSW).
- Insertion is amortized, not per-remember: a new vector lands in the flat "tail"
  (scanned) and is folded into the graph in bulk during `maintain`; recall searches
  both the graph and the tail and merges. This removes the worst case from remember.
- Deletions are tombstones; past ~10% dead, `maintain` rebuilds the graph.
- Flat → HNSW switches automatically in `maintain` when the threshold is crossed (a
  bulk build, with progress in the report).

`VectorIndex` is an enum `{ Flat, Hnsw { graph, flat_tail } }` behind one
`search(query_i8, k, ef, allow: &Bitmap) -> TopK`. Representative numbers: flat 24k×384
k=8 ~332 µs; HNSW 30k×384 ~185 µs; graph build ~1.6 ms/vector in maintain.

## 6. Graph expansion

Anchor entities come from the query (mentioned names → EntityId via the interner, plus
an explicit `entities` parameter). Expansion: depth ≤ 2 (Config), walking the out/in
arenas by range scan; it collects neighbour entities' facts (entity → facts index) and
the provenance facts of traversed edges. A candidate's weight is
`w_graph · decay^depth` (decay = 0.5), and the edge list itself goes into the result
(an agent benefits from seeing the links, not only the facts). The traversal budget is
hard: ≤ 64 entities, ≤ 256 candidate facts, ≤ 128 result edges, and ≤ 2048 scanned
posting entries (a 14k-fact hub must not turn the walk into a full decode of its list).

## 7. Fusion: hybrid recall

Input is a `RecallQuery` (see `05-api.md`): text?, vector?, tags[], entities[], time
params, k, token_budget. The pipeline reuses scratch buffers (no allocations — the
invariant):

1. Set filters: tags ∩ live ∩ as_of → an allow bitmap (if no tags, just live ∩ as_of,
   lazily).
2. Sources (each returns a ranked list ≤ 128): BM25 (if text), vector (if vector),
   graph (if anchors), time range (if range, ranked by freshness).
3. **RRF**: `score(f) = Σ_sources w_s / (60 + rank_s(f))`; weights from Config (default
   1.0 each). Simple, stable, tuning-free — no need to calibrate source scores against
   each other.
4. Recency boost: `score ·= 1 + w_rec · 2^(-(now - recorded_at)/half_life)` (default
   half_life 180 days, w_rec 0.25).
5. Dedup by revision chain: only the latest valid version of a chain (with
   `include_closed`, the whole chain marked by intervals).
6. Budget selection: greedy by score, a fact's cost estimated as len(text)/4 + 8
   overhead tokens; stop at k and token_budget.
7. Result assembly: structural (`Vec<RecalledFact>`: id, score, sources, intervals,
   edges) plus a rendered compact text block (the template is in `05-api.md`).

## Test plan

- Tokenizer: a unit-case table (ru/en/digits/emoji/CJK/empty/64B+); property:
  concatenation with a separator does not change the token multiset.
- BM25: reference scores on a mini-corpus computed by an independent script (numbers in
  testdata); idf monotonicity.
- Vectors: quantization error dot(i8) vs dot(f32) < 3% on random pairs; flat recall@10
  ≥ 0.95 vs brute-force f32 on testgen synthetics; HNSW recall@10 ≥ 0.9 (ef=64) — a gate.
- Graph: a walk with a 10k-edge hub stays within the truncation budget.
- Hybrid: scenario tests (sources agree/disagree, each source absent); property —
  result ⊆ allow-set; determinism (two recalls in a row are identical).
- All run both natively and under wasmtime (cargo test --target + a wasmtime runner in
  CI).
