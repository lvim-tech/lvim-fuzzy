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

When installed through **lvim-installer**, the native build runs automatically as
a post-install step when a Rust toolchain is present, and a prebuilt per-platform
binary is used otherwise.

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

-- One-off stateless rank (prepare + match), for callers that keep no context:
local results2 = fuzzy.filter({ "abc", "abd" }, "ab")

-- Introspection (used by :checkhealth):
fuzzy.backend() -- "native-avx2" | "native-scalar" | "lua"
fuzzy.native_loaded() -- boolean
fuzzy.library_path() -- string|nil
fuzzy.abi() -- integer (ABI this Lua expects)
```

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
