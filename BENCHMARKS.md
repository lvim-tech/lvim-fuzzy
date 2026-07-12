# lvim-fuzzy — fuzzy engine benchmark

A fair, same-input comparison of `lvim-fuzzy` (the native Rust matcher behind
`lvim-picker` and `lvim-cmp`) against the other fuzzy engines in common use. Every
engine is fed the **identical corpus** and the **identical query** for each scenario,
so the numbers reflect the matcher — not different input, caching or corpus.

## Scenarios (measured separately)

| Scenario | What it measures | Corpus | Query |
| --- | --- | --- | --- |
| **files** | fuzzy match + rank a query over a large file-path list (per-keystroke latency) | 1,000,000 paths under `$HOME` | `hel` |
| **grep · ingest** | cost to turn raw `rg` output into the engine's matchable structure | 300,000 `rg` lines | — |
| **grep · filter** | fuzzy match + rank a secondary query over the loaded grep results (the `<C-g>` filter) | 300,000 `rg` lines | `err` |

Both queries deliberately match nearly the **whole** corpus, so every engine has to
score every candidate — a worst-case throughput test, not a lucky early-out.

## Results

Wall-clock milliseconds — **lower is better**. Best in each column is **bold**.

| Engine | files (1M) | grep · ingest (300k) | grep · filter (300k) |
| --- | ---: | ---: | ---: |
| **lvim-fuzzy** — parallel / blob *(shipped default)* | **41.9** | **113.0** | **16.2** |
| lvim-fuzzy — single core / prepared | 193.4 | 113.0 | 85.0 |
| snacks.picker | 553.4 | 535.7 | 482.4 |
| fzf (`--filter`) | 1117.8 | — | 432.1 |
| mini.fuzzy | 4820.3 | 518.6 | 1174.5 |
| telescope (default sorter) | 5353.0 | 533.9 | 173.0 |

### Speed-up of the shipped engine (parallel / blob)

| vs | files | grep · filter |
| --- | ---: | ---: |
| fzf | **26.7×** | **26.7×** |
| snacks.picker | 13.2× | 29.8× |
| mini.fuzzy | 115.0× | 72.5× |
| telescope | 127.8× | 10.7× |

## Notes on fairness

- **Result materialisation.** `lvim-fuzzy` scores the full corpus every keystroke but
  only builds the **top-K** rows (`result_cap`, default 1000) — because only the visible
  window is ever shown. mini.fuzzy / telescope / snacks collect **and sort the entire
  match set**, and `fzf --filter` prints every match. All engines pay the full
  *scoring* cost measured here; the smaller output is a real architectural win of the
  blob path (it never allocates or sorts a million-entry Lua result array), not a
  measurement shortcut. Reported match counts for the files run: lvim-fuzzy top-1000,
  snacks / fzf 1,000,000 (all), telescope 411,760, mini.fuzzy 1,000,000.
- **Ingest.** For the Lua engines "ingest" is the unavoidable cost of splitting the raw
  `rg` bytes into a Lua string array and wrapping each line in the engine's item/entry
  table. `lvim-fuzzy` appends the raw bytes straight into a native blob (no per-line Lua
  object, no LuaJIT string interning), which is why it ingests ~4.6× faster. `fzf` reads
  its stdin inside its own process, so its ingest is folded into the subprocess time and
  is shown as `—`.
- **Configuration used.** telescope runs its **default** pure-Lua ngram sorter
  (`telescope-fzf-native` deliberately not installed, to measure the honest out-of-the-box
  default). snacks uses its built-in Lua matcher with `fuzzy = true`. mini.fuzzy uses
  `MiniFuzzy.filtersort`. fzf uses `fzf --filter` (single-threaded C). `lvim-fuzzy` uses
  the shipped native library (Rust + AVX2 SIMD + multi-core scan).
- **Timing.** Median of 3 runs for the fast engines (`lvim-fuzzy`, `fzf`) and the ingest
  column; a single pass for the pure-Lua match runs (deterministic, and repeating a
  multi-second pass over 1M candidates adds nothing but minutes). GC collected between
  runs.

## Why lvim-fuzzy is faster

1. **Native, not interpreted.** The scorer is a Rust affine-gap matcher; the hot loop
   never touches the LuaJIT interpreter or its garbage collector.
2. **SIMD prefilter.** A per-candidate signature (AVX2) rejects non-matches before the
   full DP scorer runs.
3. **Multi-core.** The candidate set is scanned in parallel across all cores
   (`std::thread::scope` fan-out), merged in stable index order — the single-core row
   above is the same engine with the fan-out disabled.
4. **Blob ingest.** Candidates live as raw bytes in a handle-addressed native context;
   the picker never materialises a million-string Lua pool, so there is no interning or
   GC pause while streaming a large source.

## Reproduce

The harness and the fixed corpora live under `~/.cache/lvim-bench`:

```sh
# corpus: fd over $HOME (files) + rg (grep), generated once
bash ~/.cache/lvim-bench/driver.sh        # runs fzf + the in-nvim engines
cat ~/.cache/lvim-bench/results.json      # lvim-fuzzy / mini.fuzzy / telescope / snacks
cat ~/.cache/lvim-bench/fzf.json          # fzf
```

Environment for the numbers above: 24 logical cores, ripgrep 15.1.0, fzf 0.65,
Neovim headless.
