-- lvim-fuzzy: the live configuration table.
-- Holds the defaults; setup() merges user overrides into it in place (via
-- lvim-utils.utils.merge), so every require("lvim-fuzzy.config") reader sees the
-- effective values. lvim-fuzzy has very few knobs on purpose — it is a matching
-- ENGINE (blink.lib's role), not a UI: the ranking model is fixed so results are
-- identical across the native and the pure-Lua backends.
--
---@module "lvim-fuzzy.config"

---@class LvimFuzzyConfig
---@field backend        "auto"|"native"|"lua"  Which matcher to use: auto = native .so when loadable else Lua
---@field library_path   string?                Explicit path to liblvimfuzzy.{so,dylib,dll}; nil = auto-probe
---@field max_results    integer                Hard cap on ranked results returned per query (0 = no cap)
---@field warn_on_fallback boolean              Notify once (at INFO) when the native .so is unavailable

---@type LvimFuzzyConfig
return {
    -- "auto" loads the native library when it is present AND its ABI matches, and
    -- transparently uses the byte-identical pure-Lua matcher otherwise. "native"
    -- forces the .so (errors in :checkhealth if missing); "lua" forces the fallback.
    backend = "auto",
    -- Absolute path to the shared library. nil ⇒ probe (in order): $LVIM_FUZZY_LIB,
    -- the plugin's own native/build/, then native/target/release/ (a local dev build).
    library_path = nil,
    -- Cap the materialised result list so a broad query over a huge candidate set
    -- never builds hundreds of thousands of rows on a keystroke. 0 disables the cap.
    max_results = 1000,
    -- Emit a single INFO notification the first time a match runs on the Lua
    -- fallback (so a user without a Rust toolchain knows why it is a touch slower).
    warn_on_fallback = true,
}
