-- lvim-fuzzy: :checkhealth lvim-fuzzy.
-- Reports whether the native library (the C ABI over lvim-fuzzy-core) is loaded, from where, its ABI and
-- whether the CPU runs core's AVX2 paths, validates the config, and ranks a small sample as a smoke test.
-- Read-only.
--
---@module "lvim-fuzzy.health"

local config = require("lvim-fuzzy.config")
local fuzzy = require("lvim-fuzzy")

local M = {}

--- Validate the config values, reporting each problem.
---@param h table  the vim.health reporter
local function check_config(h)
    local ok = true
    local function bad(msg)
        h.error(msg)
        ok = false
    end
    if type(config.max_results) ~= "number" or config.max_results < 0 then
        bad("config.max_results must be a number ≥ 0")
    end
    if config.library_path ~= nil and type(config.library_path) ~= "string" then
        bad("config.library_path must be a string path or nil")
    end
    if not ({ smart = true, ignore = true, respect = true })[config.case] then
        bad(("config.case must be 'smart'|'ignore'|'respect' (got %s)"):format(vim.inspect(config.case)))
    end
    if not ({ default = true, path = true, history = true })[config.scheme] then
        bad(("config.scheme must be 'default'|'path'|'history' (got %s)"):format(vim.inspect(config.scheme)))
    end
    if config.typos ~= "auto" and (type(config.typos) ~= "number" or config.typos < 0) then
        bad("config.typos must be a number ≥ 0 or 'auto'")
    end
    if ok then
        h.ok("config is valid")
    end
end

--- Entry point for `:checkhealth lvim-fuzzy`.
function M.check()
    local h = vim.health
    h.start("lvim-fuzzy")

    check_config(h)

    if not fuzzy.native_loaded() then
        h.error("native library not loaded: " .. tostring(fuzzy.load_error()))
        h.info(
            "lvim-fuzzy has no pure-Lua matcher. Build it with `sh native/build.sh` in the plugin directory "
                .. "(needs cargo; the first build fetches lvim-fuzzy-core from GitHub). lvim-pack runs it as the "
                .. "plugin's build hook."
        )
        return
    end

    h.ok(("library: %s (ABI %d)"):format(fuzzy.library_path() or "?", fuzzy.native_abi() or -1))
    if fuzzy.backend() == "native-avx2" then
        h.ok("AVX2 available — the engine's SIMD paths are in use")
    else
        h.info("no AVX2 on this CPU — the engine runs its scalar paths")
    end

    local ok, res = pcall(fuzzy.filter, { "lua/init.lua", "src/main.rs", "README.md" }, "init")
    if ok and res[1] and res[1].index == 1 then
        h.ok("a sample query ranks as expected")
    else
        h.error("a sample query failed: " .. (ok and vim.inspect(res) or tostring(res)))
    end
end

return M
