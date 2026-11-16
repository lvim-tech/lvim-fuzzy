-- lvim-fuzzy: the live configuration table.
-- Holds the defaults; setup() merges user overrides into it in place (via
-- lvim-utils.utils.merge), so every require("lvim-fuzzy.config") reader sees the
-- effective values. The search options here are the defaults of every call; a
-- consumer passes its own `opts` per call (lvim-cmp: no query syntax, ignore case).
--
---@module "lvim-fuzzy.config"

---@class LvimFuzzyConfig
---@field library_path   string?           Explicit path to liblvimfuzzy.{so,dylib,dll}; nil = auto-probe
---@field max_results    integer           Hard cap on ranked results returned per query (0 = no cap)
---@field case           "smart"|"ignore"|"respect"
---@field extended       boolean           fzf's extended query syntax
---@field exact          boolean           every term is an exact substring
---@field normalize      boolean           Latin letters match their accented forms (`cafe` → `café`)
---@field scheme         "default"|"path"|"history"
---@field typos          integer|"auto"    characters a fuzzy term may miss
---@field exact_window   integer?          widest window scored exactly; nil = the engine's default

---@type LvimFuzzyConfig
return {
    -- Absolute path to the shared library. nil ⇒ probe (in order): $LVIM_FUZZY_LIB,
    -- the plugin's own native/build/, then native/target/release/ (a local dev build).
    library_path = nil,
    -- Cap the materialised result list so a broad query over a huge candidate set
    -- never builds hundreds of thousands of rows on a keystroke. 0 disables the cap.
    max_results = 1000,
    -- Search defaults — the same as lvim-fuzzy-cli's, so a query ranks the same everywhere.
    case = "smart",
    extended = true,
    exact = false,
    normalize = true,
    scheme = "default",
    typos = 0,
    exact_window = nil,
}
