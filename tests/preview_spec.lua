-- lvim-fuzzy.preview: the Neovim side of the terminal finder's preview. A result whose query matched in
-- its PATH must still show where the query sits in the file, and the result's own line:col reads harder.

local preview = require("lvim-fuzzy.preview")

describe("lvim-fuzzy.preview: the query's terms")

it("strips the extended-search markers and drops negations and |", function()
    local t = preview.terms("'health ^lvim ls$ !test | Foo")
    eq({ "health", "lvim", "ls", "Foo" }, vim.tbl_map(function(x)
        return x.text
    end, t))
    eq({ true, true, true, false }, vim.tbl_map(function(x)
        return x.ci
    end, t))
end)

describe("lvim-fuzzy.preview: matches")

it("finds every occurrence, smart-case", function()
    local m = preview.matches({ "health.start(Health)", "no", "a health" }, "healt")
    eq(3, #m)
    eq({ row = 0, c0 = 0, c1 = 5 }, m[1])
    eq({ row = 0, c0 = 13, c1 = 18 }, m[2])
    eq({ row = 2, c0 = 2, c1 = 7 }, m[3])
    eq(0, #preview.matches({ "health" }, "Health"))
end)

describe("lvim-fuzzy.preview: show")

it("opens the file at its line and marks the query there", function()
    local f = vim.fn.tempname()
    vim.fn.writefile({ "one", "local h = health", "three health" }, f)
    preview.show({ path = f, line = 2, col = 11, query = "health" })
    eq(vim.fn.fnamemodify(f, ":p"), vim.api.nvim_buf_get_name(0))
    eq(2, vim.api.nvim_win_get_cursor(0)[1])
    local ns = vim.api.nvim_get_namespaces()["lvim-fuzzy-preview"]
    local marks = vim.api.nvim_buf_get_extmarks(0, ns, 0, -1, { details = true })
    eq(2, #marks)
    -- the one on line 2 at col 11 is the focused hit
    local focused = marks[1][4].priority == 260 and marks[1] or marks[2]
    eq(1, focused[2])
    eq(10, focused[3])
end)

it("shows non-file text in a scratch buffer", function()
    preview.show({ path = "no/such/file", text = "plain item", query = "item" })
    eq({ "plain item" }, vim.api.nvim_buf_get_lines(0, 0, -1, false))
    eq("nofile", vim.bo.buftype)
end)
