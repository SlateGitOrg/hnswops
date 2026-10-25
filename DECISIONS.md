# Decisions

Rulings made while building, in the form `what — why — cost if wrong`.

## Algorithm

- Ruling: implement HNSW from the paper (Algorithms 1, 2, 4, 5) with the diversity heuristic but without `extendCandidates` / `keepPrunedConnections` — the paper reports both as marginal on typical data and they add build cost — if wrong, recall on some hard distributions is a little lower than hnswlib at equal parameters; both are local additions to `select_neighbors`.
- Ruling: layer 0 degree `2M`, `mL = 1/ln M`, level cap 16 — paper defaults, also what hnswlib uses — if wrong, only tuning; all are in `HnswConfig`.
- Ruling: L2 is stored and reported as *squared* distance — avoids a `sqrt` per comparison and is order-equivalent — if wrong, callers who need true L2 distances must `sqrt` the reported value (documented on `Metric`).
- Ruling: no `unsafe` SIMD intrinsics; 8-accumulator loops that LLVM auto-vectorizes — keeps `#![deny(unsafe_code)]` meaningful (the only exception is `Mmap::map`) — if wrong, distance kernels are slower than hand-written AVX2/NEON (expect up to ~2x on large `d`).
- Ruling: upsert = tombstone old slot + insert new slot rather than in-place edge rewiring — readers never observe a node whose vector and edges disagree — if wrong, heavy upsert workloads accumulate tombstones faster and need more frequent vacuum.

## Deletion

- Ruling: three tiers (tombstone, local repair on delete, operator-driven vacuum) instead of immediate physical removal — tombstones keep the graph routable, local repair bounds degradation, vacuum restores full invariants at a time of the operator's choosing — if wrong, memory is held by tombstones until vacuum; `HealthReport` exposes the count so it can be alerted on.
- Ruling: local repair back-fills each repaired list to its previous length with the nearest leftovers after the heuristic — the pure heuristic shrinks lists and repeated deletions erode in-degree — if wrong, some lists carry a few less-diverse edges; measured recall after 30–40% deletion on the fixture stays ≥ 0.999.
- Ruling: expose `repair_connectivity()` publicly and count `unreachable_l0` in health — a fresh build can orphan nodes when the heuristic prunes their last in-edge (observed: a few nodes at M=12/efC=100 on the fixture) — if wrong (i.e. operators never run it), a handful of points are unfindable until the next vacuum.
- Ruling: vacuum rebuilds vector storage on the heap (drops the mmap) — compaction renumbers ids so the mapped file no longer matches — if wrong, a vacuumed index uses `4·n·d` bytes of RSS until it is snapshotted and reloaded.

## Filtering

- Ruling: metadata is a 64-bit tag mask plus the key, not arbitrary JSON — keeps per-node metadata fixed-size, persisted in `graph.bin`, and predicates branch-free — if wrong, rich metadata needs a side store keyed by the same key; the `Filter` enum is the extension point.
- Ruling: cost-based planner (graph walk ≈ `(ef/s)·M` distances vs scan ≈ `s·n + n/32`) replacing the initial fixed 5% selectivity cutoff — the fixed cutoff chose a 10k-evaluation graph walk where a 630-evaluation scan was exact — if wrong, the constants mis-price some workloads; the demo prints estimated vs measured cost so it can be recalibrated, and the 5% floor is still configurable.
- Ruling: selectivity from a deterministic strided sample of 512 nodes — deterministic plans make tests and debugging reproducible — if wrong, adversarially periodic tag layouts can fool the sample (falls back to a full count only when the sample sees no live node).

## Persistence and snapshots

- Ruling: vectors in their own mmap-able file, graph parsed into memory — vectors dominate size (`d` floats vs ~`2M` ints per node) and are read randomly, the graph needs validation anyway — if wrong, graph parse time (`O(n·M)`) dominates start-up for very large indexes; a mmap'd CSR graph would be the next step.
- Ruling: verify every graph invariant on load and treat any violation as `Error::Corrupt` — a corrupt file must not become a panic or a silently wrong index — if wrong, load is slower by one pass over the graph.
- Ruling: `manifest.json` is protected by structural validation (sizes, checksums of the data files, graph invariants checked against the config) but carries no checksum of its own — it is meant to be human-readable and diffable — if wrong, an edit to e.g. `ef_search` or `seed` in the manifest is accepted silently; `m` edits are caught by the degree checks.
- Ruling: one writer process per snapshot store, no lock file — commits are rare operator actions; atomic rename already protects readers — if wrong, two concurrent committers can race on the version number (the second rename fails with an I/O error rather than corrupting anything).
- Ruling: little-endian only; big-endian hosts are rejected at load — every supported deployment target is little-endian — if wrong, big-endian users must convert files.

## Monitoring

- Ruling: shadow queries are evaluated asynchronously from a bounded queue (overflow dropped and counted) — the serving path must never pay an `O(n·d)` scan — if wrong, under sustained overload the window under-samples; `dropped` makes that visible.
- Ruling: replay the ANN query when the index generation changed between serving and evaluation — otherwise deletes/inserts in between look like recall loss — if wrong (heavy write load), many samples are replays and measure the index at evaluation time rather than at serving time; `replayed` is reported.
- Ruling: distance-aware recall (a hit counts if no farther than the k-th exact neighbor) — equidistant ties are common in quantized data (SIFT is integer-valued) and are not errors — if wrong, recall is slightly optimistic when distinct points share the k-th distance.
- Ruling: hysteresis on status and window reset after autotune — avoids alert flapping and stale samples steering the tuner — if wrong, recovery is reported one window late.

## Migration

- Ruling: per-key write-sequence compare-and-set for backfill rows — closes the lost-update race without holding a lock while embedding — if wrong, nothing known; the cost is one `u64` per key written during the migration.
- Ruling: migration state (phase, cursor, sequence map) is in memory, not persisted — the crate has no write-ahead log; a crashed migration simply starts over (the old index and its snapshots are untouched, so serving is unaffected) — if wrong, very large migrations lose progress on crash; persisting `(phase, cursor)` plus snapshotting both indexes is on the roadmap.
- Ruling: the shipped embedder is deterministic feature hashing, not a neural model; the demo does not call Ollama — tests must be deterministic and offline (contract), and two hashing configurations behave exactly like two incompatible model versions for the purposes of a migration — if wrong, nothing about *semantic* quality is demonstrated; the `Embedder` trait is where a real model plugs in.
- Ruling: verification gate checks coverage and the new index's ANN-vs-exact recall, and reports old/new top-k overlap without gating on it — a new model is *supposed* to rank differently, so overlap is not a correctness criterion — if wrong, a semantically broken model with a structurally fine index passes the gate; callers can add a gate on overlap or on labelled relevance data.

## Data and evaluation

- Ruling: SIFT10K is fetched by a checksummed script, not committed — the TEXMEX site states no explicit license — if wrong, `make check` on a fresh clone skips the SIFT recall test (it prints why); the synthetic fixture recall tests always run.
- Ruling: commit a synthetic 10k×32 fixture (1.4 MB) with held-out queries and exact ground truth — `make check` and `make demo` must work offline — if wrong, nothing; it is generated by a committed command.
- Ruling: no parameter tuned on SIFT10K — M=16/efC=200 are textbook defaults; the bench sweeps `efSearch` and reports the whole curve — if wrong, n/a.
- Ruling: coverage is measured by `cargo llvm-cov` over the whole crate including the CLI and demo (exercised by `tests/cli.rs`), gate at 80% lines — one honest number for the whole codebase — if wrong, the CLI's share inflates or deflates the core number slightly; the per-file table in the report shows core modules individually.
- Ruling: `[profile.dev]`/`[profile.test]` use `opt-level = 1` — integration tests build 10k-point indexes and would be slow unoptimized — if wrong, slightly longer compile times.
