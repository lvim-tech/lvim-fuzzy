-- The ranking as Neovim sees it through the native library (lvim-fuzzy-core). The model itself is tested in
-- core; here: the Lua API's contract — indices, counts, the total, the query syntax, boosts, growing sets,
-- streams, scopes and the positions a menu or a list highlights.

local fuzzy = require("lvim-fuzzy")

if not fuzzy.native_loaded() then
    describe("lvim-fuzzy: the ranking")
    skip("every ranking test", "the native library is not built here: " .. tostring(fuzzy.load_error()))
    return
end

--- Rank and describe the result as the candidates themselves.
---@param cands string[]
---@param query string
---@param opts? LvimFuzzyOpts
---@return string[]
local function ranked(cands, query, opts)
    local res = fuzzy.filter(cands, query, opts)
    local out = {}
    for i, r in ipairs(res) do
        out[i] = cands[r.index]
    end
    return out
end

describe("lvim-fuzzy: what may match")

it("an empty query is the identity, in source order", function()
    eq({ "a", "b", "c" }, ranked({ "a", "b", "c" }, ""))
    local res, count, total = fuzzy.filter({ "a", "b", "c" }, "")
    eq(0, res[1].score, "nothing was scored, so nothing has a score")
    eq(3, count)
    eq(3, total)
end)

it("a candidate the query is not a subsequence of does not match", function()
    eq({ "fbz", "foobar" }, ranked({ "foobar", "fbz", "zzz" }, "fb"))
end)

it("smart case: lower case folds, an upper-case letter is respected", function()
    eq(2, #ranked({ "FooBar", "foobar" }, "fb"))
    eq({ "FooBar" }, ranked({ "FooBar", "foobar" }, "FB"))
    eq(2, #ranked({ "FooBar", "foobar" }, "FB", { case = "ignore" }))
end)

it("spaces are AND, and the fzf syntax works", function()
    local cands = { "lua/test.lua", "lua/init.lua", "src/init.rs" }
    eq({ "lua/init.lua" }, ranked(cands, "init lua"))
    eq({ "lua/init.lua" }, ranked(cands, "lua !test"))
    eq({ "src/init.rs" }, ranked(cands, ".rs$"))
    eq({ "src/init.rs" }, ranked(cands, "^src"))
    eq({}, ranked(cands, "lua !test", { extended = false }), "without the syntax the query is one needle")
end)

it("Latin letters match their accented forms; Cyrillic folds case", function()
    eq({ "café" }, ranked({ "café", "tea" }, "cafe"))
    eq({ "Пиши" }, ranked({ "Пиши", "друго" }, "пи"))
end)

it("the total counts every match, the cap only what is returned", function()
    local cands = {}
    for i = 1, 50 do
        cands[i] = "item" .. i
    end
    local max = require("lvim-fuzzy.config").max_results
    require("lvim-fuzzy.config").max_results = 10
    local res, count, total = fuzzy.filter(cands, "item")
    require("lvim-fuzzy.config").max_results = max
    eq(10, #res)
    eq(10, count)
    eq(50, total)
end)

describe("lvim-fuzzy: what ranks above what")

it("a match at the start beats one in the middle", function()
    eq({ "bar", "abar" }, ranked({ "abar", "bar" }, "b"))
end)

it("a match after a boundary character beats one inside a word", function()
    eq({ "my_bar", "mybar" }, ranked({ "mybar", "my_bar" }, "bar"))
    eq({ "a/bar", "axbar" }, ranked({ "axbar", "a/bar" }, "bar"))
end)

it("consecutive characters beat scattered ones", function()
    eq({ "abc", "axbxc" }, ranked({ "axbxc", "abc" }, "abc"))
end)

it("the order is total: score desc, then shorter, then earlier", function()
    local cands = {}
    for i = 1, 60 do
        cands[#cands + 1] = ("x"):rep(i % 7) .. "ab" .. ("y"):rep(i % 5)
    end
    local res = fuzzy.filter(cands, "ab")
    truthy(#res > 10, "precondition: plenty of matches to order (" .. #res .. ")")
    for k = 2, #res do
        local prev, cur = res[k - 1], res[k]
        if prev.score == cur.score then
            local lp, lc = #cands[prev.index], #cands[cur.index]
            if lp == lc then
                truthy(prev.index < cur.index, ("equal score and length at %d: index went backwards"):format(k))
            else
                truthy(lp < lc, ("equal score at %d: the longer candidate came first"):format(k))
            end
        else
            truthy(prev.score > cur.score, ("scores out of order at %d"):format(k))
        end
    end
end)

it("boosts are added before ranking; ties keep the engine's order", function()
    local cands = { "abc", "abd", "abe" }
    local ctx = fuzzy.prepare(cands)
    local plain = fuzzy.match("ab", ctx)
    eq({ 1, 2, 3 }, vim.tbl_map(function(r)
        return r.index
    end, plain))
    local boosted = fuzzy.match("ab", ctx, { 0, 0, 5 })
    eq({ 3, 1, 2 }, vim.tbl_map(function(r)
        return r.index
    end, boosted))
    eq(plain[3].score + 5, boosted[1].score)
    fuzzy.free(ctx)
end)

describe("lvim-fuzzy: contexts")

it("the prepared match and the one-off filter agree", function()
    local cands = { "init.lua", "lvim/fuzzy.lua", "my_name", "FooBar", "x.y", "" }
    local ctx = fuzzy.prepare(cands)
    for _, q in ipairs({ "fb", "init", "my_n", "x", "lua" }) do
        eq(fuzzy.filter(cands, q), (fuzzy.match(q, ctx)), "query " .. q)
    end
    fuzzy.free(ctx)
end)

it("an empty candidate keeps its index, so results stay parallel to the list", function()
    local cands = { "", "abc", "", "xbc" }
    local ctx = fuzzy.prepare(cands)
    eq(4, ctx.n)
    eq({ 2, 4 }, vim.tbl_map(function(r)
        return r.index
    end, (fuzzy.match("bc", ctx))))
    fuzzy.free(ctx)
end)

it("a context can grow, and typing on narrows to the same result as a fresh search", function()
    local ctx = fuzzy.prepare({ "alpha", "beta" })
    local before = fuzzy.match("a", ctx)
    fuzzy.append(ctx, { "gamma_a", "delta" })
    local after = fuzzy.match("a", ctx)
    truthy(#after > #before, "the new candidates have to be reachable")
    local all = { "alpha", "beta", "gamma_a", "delta" }
    eq(fuzzy.filter(all, "al"), (fuzzy.match("al", ctx)), "narrowed from 'a'")
    fuzzy.free(ctx)
    fuzzy.free(ctx) -- twice is harmless
    eq(0, select(2, fuzzy.match("a", ctx)), "a freed context matches nothing")
end)

it("match_async delivers on a later tick", function()
    local ctx = fuzzy.prepare({ "one", "two", "three" })
    local got
    fuzzy.match_async("t", ctx, function(res, count, total)
        got = { #res, count, total }
    end)
    eq(nil, got, "not inside the call")
    vim.wait(1000, function()
        return got ~= nil
    end)
    eq({ 2, 2, 2 }, got)
    fuzzy.free(ctx)
end)

describe("lvim-fuzzy: blob contexts")

it("a stream is the same set however it is cut; empty lines are skipped", function()
    local text = "src/main.rs\nlua/init.lua\n\nREADME.md\nsrc/lib.rs"
    local blob = fuzzy.blob_new()
    for i = 1, #text, 3 do
        fuzzy.blob_append(blob, text:sub(i, i + 2))
    end
    eq(3, fuzzy.blob_count(blob), "the last line waits for the flush")
    eq(4, fuzzy.blob_flush(blob))
    eq("src/lib.rs", fuzzy.blob_text(blob, 4))
    eq(nil, fuzzy.blob_text(blob, 5))
    local got
    fuzzy.blob_match(blob, "src", function(res, count, total)
        got = { idx = vim.tbl_map(function(r)
            return r.index
        end, res), count = count, total = total }
    end)
    vim.wait(1000, function()
        return got ~= nil
    end)
    eq({ idx = { 4, 1 }, count = 2, total = 2 }, got)
    fuzzy.blob_free(blob)
    fuzzy.blob_free(blob)
    eq(0, fuzzy.blob_count(blob))
end)

it("a grep result is searched by the scope it is given", function()
    local blob = fuzzy.blob_new()
    fuzzy.blob_append(blob, "lua/health.lua:3:local x\nlua/init.lua:9:health()\n")
    local function run(scope)
        local got
        fuzzy.blob_match(blob, "health", function(res)
            got = vim.tbl_map(function(r)
                return r.index
            end, res)
        end, { scope = scope })
        vim.wait(1000, function()
            return got ~= nil
        end)
        return got
    end
    eq({ 2 }, run("line"))
    eq({ 1 }, run("path"))
    eq(2, #run("all"))
    fuzzy.blob_free(blob)
end)

describe("lvim-fuzzy: the positions a menu highlights")

it("the columns are ascending, inside the candidate, and spell the query", function()
    for _, case in ipairs({
        { "FooBar", "fb" },
        { "foobar", "fb" },
        { "src/lvim-fuzzy/init.lua", "lfinit" },
        { "my_long_name", "mln" },
    }) do
        local cand, query = case[1], case[2]
        local pos = fuzzy.positions(cand, query)
        truthy(pos, "no positions for " .. query .. " in " .. cand)
        eq(#query, #pos, "one column per query byte")
        local spelled = {}
        for i, col in ipairs(pos) do
            truthy(col >= 1 and col <= #cand, "column " .. col .. " is outside " .. cand)
            if i > 1 then
                truthy(col > pos[i - 1], "columns must ascend")
            end
            spelled[#spelled + 1] = cand:sub(col, col):lower()
        end
        eq(query:lower(), table.concat(spelled), "the highlighted bytes have to BE the query")
    end
end)

it("a multibyte character is every one of its bytes", function()
    eq({ 1, 2, 3, 4 }, fuzzy.positions("Пиши", "пи"))
end)

it("there are no positions where there is no match", function()
    eq(nil, fuzzy.positions("abc", "zzz"))
    eq(nil, fuzzy.positions("abc", ""), "an empty query highlights nothing")
    eq(nil, fuzzy.positions("", "a"))
end)

it("highlights are character indices of every occurrence of every term", function()
    eq({ 0, 1, 3, 4 }, fuzzy.highlights("ab ab", "ab"))
    eq({ 0, 1, 4, 5 }, fuzzy.highlights("abcdef", "ab ef"))
    eq(nil, fuzzy.highlights("abcdef", "ab zz"))
    eq({ 0, 1 }, fuzzy.highlights("Пиши", "пи"))
end)
