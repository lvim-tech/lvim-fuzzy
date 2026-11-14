-- lvim-fuzzy.preview: the Neovim side of the lvim-fuzzy terminal finder's preview.
--
-- The finder (github.com/lvim-tech/lvim-fuzzy-cli) can show its preview in a REAL Neovim: it runs one
-- `nvim --embed` (marked with g:lvim_fuzzy_preview) and calls `show()` for each focused result. Everything
-- about how that preview looks lives HERE, in Neovim — the finder only passes the target and the query:
--   • opens the file at line:col (a scratch buffer with the item text when it is not a file);
--   • marks EVERY match of the query in the buffer, and the one the result points at (its line:col)
--     harder — the same contract as lvim-picker's preview (`preview_match` / `preview_match_focus`), so a
--     result that matched in its PATH still shows where the query sits in the file;
--   • wipes the previously previewed buffer (a preview must not pile up buffers), later, so callbacks
--     scheduled for it (LSP attach and the like) run before it disappears.
--
---@module "lvim-fuzzy.preview"

local api = vim.api

local M = {}

local NS = api.nvim_create_namespace("lvim-fuzzy-preview")

--- Highlight groups (lvim-utils defines them; plain fallbacks otherwise).
M.hl = {
    match = "LvimUiMsgAreaMatch",
    focus = "LvimUiPeekTitle",
}

---@param name string
---@param fallback string
---@return string
local function group(name, fallback)
    return vim.fn.hlexists(name) == 1 and name or fallback
end

--- The query's literal terms: fzf extended-search markers stripped (`'x`, `^x`, `x$`), negations (`!x`)
--- and `|` dropped. Smart case per term: no uppercase letter → case-insensitive.
---@param query string
---@return { text: string, ci: boolean }[]
function M.terms(query)
    local out = {}
    for tok in (query or ""):gmatch("%S+") do
        if tok ~= "|" and tok:sub(1, 1) ~= "!" then
            local t = tok:gsub("^['^]+", ""):gsub("[$']+$", "")
            if t ~= "" then
                out[#out + 1] = { text = t, ci = not t:find("%u") }
            end
        end
    end
    return out
end

--- Every match of the query's terms in `lines`, as 0-based byte ranges.
---@param lines string[]
---@param query string
---@return { row: integer, c0: integer, c1: integer }[]
function M.matches(lines, query)
    local out = {}
    for _, term in ipairs(M.terms(query)) do
        local needle = term.ci and term.text:lower() or term.text
        for i, line in ipairs(lines) do
            local hay = term.ci and line:lower() or line
            local at = 1
            while true do
                local s0, e0 = hay:find(needle, at, true)
                if not s0 then
                    break
                end
                out[#out + 1] = { row = i - 1, c0 = s0 - 1, c1 = e0 }
                at = e0 + 1
            end
        end
    end
    return out
end

local wipe_pending = {}

--- Wipe `buf` a little later unless it is shown again or was modified.
---@param buf integer
local function wipe_later(buf)
    wipe_pending[buf] = true
    vim.defer_fn(function()
        if not wipe_pending[buf] then
            return
        end
        wipe_pending[buf] = nil
        if
            api.nvim_buf_is_valid(buf)
            and buf ~= api.nvim_get_current_buf()
            and not vim.bo[buf].modified
            and vim.fn.bufwinnr(buf) == -1
        then
            pcall(api.nvim_buf_delete, buf, { force = false })
        end
    end, 1500)
end

--- Show one result.
---@param opts { path: string, line?: integer, col?: integer, text?: string, query?: string }
function M.show(opts)
    local path, line, col = opts.path or "", opts.line or 0, opts.col or 0
    local prev = api.nvim_get_current_buf()
    local buf
    if path ~= "" and vim.fn.filereadable(path) == 1 then
        local existing = vim.fn.bufnr(vim.fn.fnamemodify(path, ":p"))
        if existing ~= -1 and existing == prev then
            buf = prev -- the same file again (a new line or a new query): no reload
        elseif pcall(vim.cmd, "silent keepalt edit " .. vim.fn.fnameescape(path)) then
            buf = api.nvim_get_current_buf()
        end
    end
    if not buf then
        buf = api.nvim_create_buf(false, true)
        vim.bo[buf].bufhidden = "wipe"
        api.nvim_buf_set_lines(buf, 0, -1, false, vim.split(opts.text or "", "\n", { plain = true }))
        api.nvim_set_current_buf(buf)
    end
    wipe_pending[buf] = nil
    if prev ~= buf then
        wipe_later(prev)
    end
    local n = api.nvim_buf_line_count(buf)
    if line > 0 then
        line = math.min(line, n)
        pcall(api.nvim_win_set_cursor, 0, { line, math.max(col - 1, 0) })
        vim.cmd("normal! zz")
    else
        pcall(api.nvim_win_set_cursor, 0, { 1, 0 })
    end
    -- where the query is: every match, the focused one harder
    api.nvim_buf_clear_namespace(buf, NS, 0, -1)
    local lines = api.nvim_buf_get_lines(buf, 0, -1, false)
    for _, r in ipairs(M.matches(lines, opts.query or "")) do
        local focused = line > 0 and r.row == line - 1 and col > 0 and r.c0 <= col - 1 and col - 1 < r.c1
        pcall(api.nvim_buf_set_extmark, buf, NS, r.row, r.c0, {
            end_col = r.c1,
            hl_group = focused and group(M.hl.focus, "IncSearch") or group(M.hl.match, "Search"),
            priority = focused and 260 or 250,
        })
    end
end

return M
