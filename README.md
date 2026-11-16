# lvim-fuzzy

The fuzzy search **engine** of the lvim-tech set in Neovim. `lvim-picker` and
`lvim-cmp` both rank through it.

The engine is [lvim-fuzzy-core](https://github.com/lvim-tech/lvim-fuzzy-core),
in Rust — the same one the terminal finder
[lvim-fuzzy-cli](https://github.com/lvim-tech/lvim-fuzzy-cli) runs on, so a query
ranks the same in the picker, in completion and in the terminal. This plugin is
the Neovim side: a thin C ABI over core (`native/`, a shared library loaded
**in-process** with LuaJIT FFI) and the Lua API on top.

The Lua↔native boundary is **coarse**: a candidate set lives in a native
context, filled once (`prepare`, `append`) or straight from a process's output
(`blob_*`); every keystroke is one FFI call carrying only the query and
returning the ranked indices. While the query only grows, the next search reads
only the previous matches.

> **Breaking (ABI 7).** The native library is now required — the pure-Lua
> matcher is gone — and queries take fzf's syntax with smart case by default
> (see [Queries](#queries)). An older library (ABI 1–6) is refused;
> `:checkhealth lvim-fuzzy` says how to rebuild.

## Installation

Install with the lvim-tech installer (**lvim-installer**) or Neovim's native
[`vim.pack`](https://neovim.io/doc/user/pack.html), and build the library (needs
a Rust toolchain; the first build fetches lvim-fuzzy-core from GitHub, pinned to
a commit):

```sh
sh native/build.sh
```

The build runs automatically when the plugin's spec DECLARES it — the loader has
no way to guess that a plugin wants compiling:

```lua
vim.pack.add({
    {
        src = "https://github.com/lvim-tech/lvim-fuzzy",
        build = "sh native/build.sh",
        built = function(ctx)
            return vim.fn.filereadable(ctx.dir .. "/native/build/liblvimfuzzy.so") == 1
        end,
    },
})
```

With that declared, the loader runs the build on install and again whenever the
plugin changes, and sweeps for a missing library on later starts.

## Queries

The defaults are lvim-fuzzy-cli's (and fzf's):

| query        | matches                                            |
| ------------ | -------------------------------------------------- |
| `conf ui`    | both terms, in any order (a space is AND)          |
| `'exact`     | the exact substring                                |
| `^src`       | starts with `src`                                  |
| `.lua$`      | ends with `.lua`                                   |
| `!test`      | does not contain `test`                            |
| `lua \| rs`  | either                                             |
| `fb` / `FB`  | smart case: lower case folds, a capital is kept    |
| `cafe`       | `café` too (Latin letters match accented forms)    |

Every search call takes an optional `opts` table over the config, e.g.
lvim-cmp passes `{ extended = false, case = "ignore" }` (the typed word is one
needle, in any case).

## Setup

Calling `setup()` is optional — the defaults work. The full default config:

```lua
require("lvim-fuzzy").setup({
    -- Absolute path to liblvimfuzzy.{so,dylib,dll}. nil auto-probes: $LVIM_FUZZY_LIB,
    -- then native/build/, then native/target/release/.
    library_path = nil,
    -- Cap on the materialised result list per query (0 = no cap); the total that
    -- matched is still counted.
    max_results = 1000,
    -- Search defaults of every call (a call's `opts` overrides them).
    case = "smart", -- "smart" | "ignore" | "respect"
    extended = true, -- fzf's extended syntax
    exact = false, -- every term an exact substring
    normalize = true, -- `cafe` finds `café`
    scheme = "default", -- "default" | "path" | "history"
    typos = 0, -- characters a fuzzy term may miss, or "auto"
    exact_window = nil, -- widest window scored exactly (nil = the engine's default)
})
```

## API

```lua
local fuzzy = require("lvim-fuzzy")

-- A context once per candidate set; `opts` (optional) is what match uses by default.
-- Every candidate is kept (an empty one too), so indices are indices into the list.
local ctx = fuzzy.prepare({ "getClient", "config_set", "buffer.line" })

-- Rank on every keystroke — one FFI call.
-- results[k] = { index = <1-based>, score = <integer> }; total = every match (before the cap)
local results, count, total = fuzzy.match("gc", ctx)

-- Per-candidate score offsets folded in before the ranking (lvim-cmp's boosts);
-- ties keep the engine's order.
fuzzy.match("gc", ctx, { 0, 50, 0 })

-- Grow the set: only the new candidates cross.
fuzzy.append(ctx, { "getConfig", "get_client" })

-- The same on a later tick (a newer request supersedes it).
fuzzy.match_async("gc", ctx, function(results, count, total) end)

-- Release the native memory now (otherwise the GC does it).
fuzzy.free(ctx)

-- One-off rank without a kept context.
local r = fuzzy.filter({ "abc", "abd" }, "ab")

-- What to highlight in ONE candidate (visible rows only); nil on a non-match.
fuzzy.positions("getClient", "gc") -- 1-based BYTE columns of the alignment (a menu)
fuzzy.highlights("ab ab", "ab") -- 0-based CHARACTER indices, every occurrence (a list)

-- Introspection (used by :checkhealth):
fuzzy.backend() -- "native-avx2" | "native-scalar" | "none"
fuzzy.native_loaded() -- boolean
fuzzy.load_error() -- why not, or nil
fuzzy.library_path() -- string|nil
fuzzy.abi(), fuzzy.native_abi() -- 7, the loaded library's ABI
```

`has_append()`, `has_chunk()`, `has_par()`, `has_blob()` and `has_boost()` from
ABI 1–6 remain and say whether the library is loaded: every feature is in ABI 7.

### Blob contexts (streaming, no per-candidate Lua strings)

For a consumer streaming a HUGE candidate set (a file picker over millions of
paths), the producer's raw stdout **bytes** go straight to a native context,
which splits them on `\n` (empty lines skipped); the consumer materialises only
the rows it shows.

```lua
local b = fuzzy.blob_new() -- nil when the library is not loaded
fuzzy.blob_append(b, raw_stdout_chunk) -- any chunking; returns the count
fuzzy.blob_flush(b) -- a final line with no trailing \n
fuzzy.blob_count(b)

-- Rank on a scheduled tick. `opts.scope` searches a grep result (`path:line:text`)
-- by its "line", its "path", or its "file"'s content (every term exact).
fuzzy.blob_match(b, "src", function(results, count, total)
    local text = fuzzy.blob_text(b, results[1].index)
end, { scope = "line" })

fuzzy.blob_free(b)
```

One million paths load in ~120 ms and a query over them takes 13–35 ms
(AMD Ryzen 9 3900XT, every core). The engine's own benchmark against frizbee is
in [lvim-fuzzy-core](https://github.com/lvim-tech/lvim-fuzzy-core#benchmark).

## Ranking

A total, deterministic order: score descending, then the shorter candidate,
then the earlier one. The scoring model is lvim-fuzzy-core's
([docs/scoring.md](https://github.com/lvim-tech/lvim-fuzzy-core/blob/main/docs/scoring.md))
and is tested there.

## Tests

```sh
make test                           # the Lua API, in a headless nvim
cd native && cargo test             # the C ABI layer
```

## Health

```
:checkhealth lvim-fuzzy
```

reports whether the library is loaded (and why not), its path and ABI, whether
the CPU runs the engine's AVX2 paths, config validity, and a sample query.

## License

BSD-3-Clause © lvim-tech.
