# lvim-fuzzy

The shared native fuzzy-matching **engine** for the lvim-tech set. `lvim-cmp` and
`lvim-picker` both rank through it, so they order results one identical way.

The hot path is a small Rust shared library loaded **in-process** via LuaJIT FFI:
an affine-gap local-alignment scorer with first-char / word-boundary / camelCase
/ consecutive-run bonuses folded into the score, an **inter-candidate AVX2 SIMD**
tile that ranks 16 candidates per instruction, an O(1) byte-signature prefilter,
and a stable O(n) radix sort. When the library is absent (no Rust toolchain) a
**byte-identical pure-Lua fallback** runs, so results never change depending on
the backend.

The Lua↔native boundary is **coarse**: the candidate set is uploaded once per
completion/pick (`prepare`), then every keystroke is a single FFI call carrying
only the query and returning sorted indices + scores — never a call per candidate.

## Highlights

- **Fast.** On our benchmark (below) it beats a leading Rust SIMD fuzzy matcher
  on every tested query, at both 5k and 20k candidates.
- **Deterministic.** A total order — score descending, then shorter candidate,
  then source index ascending. Identical inputs always rank identically; no
  arbitrary equal-score reshuffling.
- **Honest fallback.** The pure-Lua matcher is a byte-exact mirror of the native
  scorer; `:checkhealth lvim-fuzzy` runs a live diff of the two.
- **Zero per-keystroke allocation** on the native path; one FFI call per keystroke.

## Installation

Install with the lvim-tech installer (**lvim-installer**) or Neovim's native
[`vim.pack`](https://neovim.io/doc/user/pack.html):

```lua
vim.pack.add({ "https://github.com/lvim-tech/lvim-fuzzy" })
```

Then, optionally, build the native library for full speed (needs a Rust
toolchain; without it the pure-Lua fallback runs):

```sh
sh native/build.sh
```

The build runs automatically when the plugin's spec DECLARES it — the loader has
no way to guess that a plugin wants compiling, so an empty spec installs the Lua
side and nothing else (which is how a machine with a Rust toolchain still ends up
on the pure-Lua matcher):

```lua
{
    build = "sh native/build.sh",
    -- Self-healing: the marker records the commit a build ran for, so a build that
    -- succeeded without producing a usable library would still look current. This
    -- says what "built" actually means.
    built = function(ctx)
        return vim.fn.filereadable(ctx.dir .. "/native/build/liblvimfuzzy.so") == 1
    end,
}
```

With that declared, the loader runs the build on install and again whenever the
plugin changes, and sweeps for a missing library on later starts. Without a Rust
toolchain the build fails harmlessly and the pure-Lua fallback runs.

## Setup

Calling `setup()` is optional — the defaults work. The full default config:

```lua
require("lvim-fuzzy").setup({
    -- "auto" loads the native library when present and ABI-compatible, and uses
    -- the byte-identical pure-Lua matcher otherwise. "native" requires the .so
    -- (error in :checkhealth if missing); "lua" forces the fallback.
    backend = "auto",
    -- Absolute path to liblvimfuzzy.{so,dylib,dll}. nil auto-probes: $LVIM_FUZZY_LIB,
    -- then native/build/, then native/target/release/.
    library_path = nil,
    -- Cap on the materialised result list per query (0 = no cap), so a broad query
    -- over a huge candidate set never builds hundreds of thousands of rows.
    max_results = 1000,
    -- Emit one INFO notification the first time a match runs on the Lua fallback.
    warn_on_fallback = true,
})
```

## API

```lua
local fuzzy = require("lvim-fuzzy")

-- Build a context once per fixed candidate set (a list of strings).
local ctx = fuzzy.prepare({ "getClient", "config_set", "buffer.line" })

-- Rank against a query on every keystroke — one FFI call.
-- results[k] = { index = <1-based index into candidates>, score = <integer> }
local results, count = fuzzy.match("gc", ctx)

-- Grow the pool WITHOUT rebuilding it: only the new candidates are scanned.
-- match() then ranks the whole grown set; result is identical to prepare(old ++ new).
fuzzy.append(ctx, { "getConfig", "get_client" })

-- Optionally release a large context's native memory when done (harmless to skip):
fuzzy.free(ctx)

-- One-off stateless rank (prepare + match), for callers that keep no context:
local results2 = fuzzy.filter({ "abc", "abd" }, "ab")

-- Matched-char byte columns (1-based, ascending) of a query inside ONE candidate —
-- the exact alignment the scorer credits, for highlighting a menu's VISIBLE rows.
-- Pure Lua on every backend (identical result either way); nil on a non-match.
local cols = fuzzy.positions("getClient", "gc")

-- Introspection (used by :checkhealth):
fuzzy.backend() -- "native-avx2" | "native-scalar" | "lua"
fuzzy.native_loaded() -- boolean
fuzzy.library_path() -- string|nil
fuzzy.abi() -- integer (highest ABI this Lua understands)
fuzzy.has_append() -- boolean (library supports incremental append, ABI >= 2)
fuzzy.has_blob() -- boolean (library supports blob ingestion, ABI >= 5)
```

### Blob ingestion (streaming, no per-candidate Lua strings)

For a consumer streaming a HUGE candidate set (e.g. a file picker over millions
of paths), materialising every candidate as a Lua string + table is the
bottleneck — a multi-second load of LuaJIT string-intern-table growth and GC
pauses over the pool. The blob API removes the Lua-side pool entirely: the
producer's raw stdout **bytes** go straight to a native context (which splits on
`\n` and stores each line as a candidate), and the consumer materialises only the
visible / top-K rows on demand via `blob_text`. Each context is a HANDLE in a
native registry (not the single active prepare/match slot), so a huge pool can
never be clobbered by another consumer preparing in between.

```lua
local b = fuzzy.blob_new() -- nil when the library predates ABI 5 (use the string path)
fuzzy.blob_append(b, raw_stdout_chunk) -- raw bytes; split on \n natively; returns the count
fuzzy.blob_flush(b) -- flush a final line with no trailing \n (call once at end of stream)
fuzzy.blob_count(b) -- current candidate count

-- Rank asynchronously (parallel scan, delivered on a scheduled tick — never blocks typing).
-- results[k] = { index = <1-based native index>, score }
fuzzy.blob_match(b, "src", function(results, count)
    local text = fuzzy.blob_text(b, results[1].index) -- materialise a row's text on demand
end)

fuzzy.blob_free(b) -- release the native memory (recommended on close)
```

Byte-identical to preparing the same lines the offsets-based way (proven by the
`blob_equals_prepare_whole` Rust test over every chunk boundary). Requires the
native backend at ABI ≥ 5; `blob_new()` returns nil otherwise, and the consumer
keeps its per-string path (gate on `fuzzy.has_blob()`).

## Incremental append

For a growing pool (e.g. a picker streaming results), `append` extends a prepared
context **in place** — it builds only the new candidates' byte offsets, 32-bit
signatures and lengths; the existing entries are never re-scanned. `match(ctx)`
then ranks the whole grown set, and the result is **byte-identical** to
`prepare(old ++ new)`.

The cost is proportional to the number of NEW candidates, not the pool size —
appending 5,000 candidates is ~constant regardless of how large the pool already
is, whereas a full re-prepare scales with the total:

| base pool | `append(+5000)` | full re-prepare | saving |
|---|---|---|---|
| 200,000 | 0.32 ms | 7.3 ms | ~23× |
| 1,500,000 | 0.39 ms | 51.8 ms | ~134× |

(Requires the native backend at ABI ≥ 2; without it — a pure-Lua backend, or an
older library — `append` still produces identical results, falling back to a full
re-prepare on the next match.)

## Backends & correctness

| backend | when | notes |
|---|---|---|
| `native-avx2` | .so present, CPU has AVX2 | inter-candidate SIMD tile (fastest) |
| `native-scalar` | .so present, no AVX2 | scalar affine-gap DP |
| `lua` | .so absent / forced | byte-exact pure-Lua mirror |

The SIMD and scalar native paths are proven byte-identical by an in-crate test
(`simd_matches_scalar`); the native and Lua backends are proven identical by the
health parity diff. So every path yields the **same ranking**.

## Benchmark

Same machine, same data, prepared per-keystroke latency (microseconds), vs a
leading Rust SIMD fuzzy matcher used as the reference. Lower is better; **bold**
= faster than the reference.

| query | matches | lvim-fuzzy | reference | speedup |
|---|---|---|---|---|
| `f` (5k) | 1554 | **39** | 70 | 1.8× |
| `co` (5k) | 879 | **61** | 65 | 1.1× |
| `buf` (5k) | 632 | **32** | 61 | 1.9× |
| `conf` (5k) | 28 | **11** | 33 | 3.1× |
| `handler` (5k) | 596 | **37** | 89 | 2.4× |
| `asyncqueue` (5k) | 5 | **3** | 31 | 9.6× |
| `f` (20k) | 6127 | **158** | 271 | 1.7× |
| `co` (20k) | 3556 | **233** | 269 | 1.2× |
| `buf` (20k) | 2428 | **132** | 238 | 1.8× |
| `handler` (20k) | 2436 | **150** | 361 | 2.4× |
| `asyncqueue` (20k) | 23 | **11** | 118 | 10.8× |

lvim-fuzzy wins every tested query at both sizes (1.1×–10.8×). All backends stay
far under the ~16 ms keystroke budget.

## Health

```
:checkhealth lvim-fuzzy
```

reports the active backend and library path, whether the .so's ABI matches,
whether AVX2 is available, config validity, and a live byte-for-byte parity diff
of the native and Lua matchers.

## License

BSD-3-Clause © lvim-tech.
