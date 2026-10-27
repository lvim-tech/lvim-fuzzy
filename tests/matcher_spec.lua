-- The ranking, and the claim the pure-Lua matcher makes about itself.
--
-- fallback.lua opens by saying it is "a BYTE-EXACT mirror of the native Rust scorer … its
-- ranking is indistinguishable from the native backend — a consumer never sees different
-- results depending on whether the binary is present". That is a strong promise, and on a
-- machine where the .so is built it is checkable: run both over the same candidates and
-- demand the same (index, score) sequence. Below that sit the properties the ranking itself
-- rests on -- the total order, the bonuses, and what may not match at all.

local fuzzy = require("lvim-fuzzy")
local fallback = require("lvim-fuzzy.fallback")
local config = require("lvim-fuzzy.config")

--- Rank with the pure-Lua matcher and describe the result as "candidate(score)" strings, which
--- is what these assertions are about.
---@param cands string[]
---@param query string
---@param cap? integer
---@return string[]
local function ranked(cands, query, cap)
    local res = fallback.match(cands, query, cap or 0)
    local out = {}
    for i, r in ipairs(res) do
        out[i] = cands[r.index]
    end
    return out
end

describe("lvim-fuzzy: what may match")

it("an empty query is the identity, in source order", function()
    eq({ "a", "b", "c" }, ranked({ "a", "b", "c" }, ""))
    local res = fallback.match({ "a", "b", "c" }, "", 0)
    eq(0, res[1].score, "nothing was scored, so nothing has a score")
end)

it("a candidate the query is not a subsequence of does not match", function()
    eq({ "fbz", "foobar" }, ranked({ "foobar", "fbz", "zzz" }, "fb"))
end)

it("a candidate shorter than the query cannot match", function()
    eq({}, ranked({ "a", "" }, "ab"))
end)

it("matching folds case", function()
    eq(2, #ranked({ "FooBar", "foobar" }, "fb"))
    eq(2, #ranked({ "FOOBAR", "foobar" }, "FB"), "the query folds too")
end)

it("the cap limits the results, not the ranking", function()
    local all = ranked({ "ab", "abc", "abcd" }, "ab")
    eq(3, #all)
    local capped = ranked({ "ab", "abc", "abcd" }, "ab", 2)
    eq(2, #capped)
    eq({ all[1], all[2] }, capped, "a cap takes the TOP results, not the first ones found")
end)

describe("lvim-fuzzy: what ranks above what")

it("a match at the start beats one in the middle", function()
    eq({ "bar", "abar" }, ranked({ "abar", "bar" }, "b"))
end)

it("a match after a boundary character beats one inside a word", function()
    eq({ "my_bar", "mybar" }, ranked({ "mybar", "my_bar" }, "bar"))
    eq({ "a/bar", "axbar" }, ranked({ "axbar", "a/bar" }, "bar"))
end)

it("a camel-case step beats a plain middle character", function()
    eq({ "FooBar", "fooxbar" }, ranked({ "fooxbar", "FooBar" }, "fbar"))
end)

it("consecutive characters beat scattered ones", function()
    eq({ "abc", "axbxc" }, ranked({ "axbxc", "abc" }, "abc"))
end)

it("the order is total: score desc, then shorter, then earlier", function()
    -- A set built to produce ties: many candidates with the same shape and length.
    local cands = {}
    for i = 1, 60 do
        cands[#cands + 1] = ("x"):rep(i % 7) .. "ab" .. ("y"):rep(i % 5)
    end
    local res = fallback.match(cands, "ab", 0)
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

describe("lvim-fuzzy: the positions a menu highlights")

it("the columns are ascending, inside the candidate, and spell the query", function()
    for _, case in ipairs({
        { "FooBar", "fb" },
        { "foobar", "fb" },
        { "src/lvim-fuzzy/init.lua", "lfinit" },
        { "my_long_name", "mln" },
    }) do
        local cand, query = case[1], case[2]
        local pos = fallback.positions(cand, query)
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

it("there are no positions where there is no match", function()
    eq(nil, fallback.positions("abc", "zzz"))
    eq(nil, fallback.positions("abc", ""), "an empty query highlights nothing")
    eq(nil, fallback.positions("", "a"))
end)

it("the public positions() answers the same on every backend", function()
    -- init.lua computes positions in Lua whatever the backend is, deliberately; this states it.
    eq(fallback.positions("FooBar", "fb"), fuzzy.positions("FooBar", "fb"))
end)

describe("lvim-fuzzy: the two backends are one matcher")

--- Candidates shaped like the things this engine actually ranks: paths, identifiers, words.
---@return string[]
local function corpus()
    local words = { "init", "lvim", "fuzzy", "match", "score", "Foo", "Bar", "my_name", "a-b", "x.y" }
    local out = {}
    for i = 1, #words do
        for j = 1, #words do
            out[#out + 1] = words[i] .. "/" .. words[j] .. ".lua"
            out[#out + 1] = words[i] .. "_" .. words[j]
            out[#out + 1] = words[i] .. words[j]:upper()
        end
    end
    out[#out + 1] = ""
    return out
end

it("native and pure-Lua rank identically, score for score", function()
    if not fuzzy.native_loaded() then
        skip("backend parity", "the native library is not built on this machine")
        return
    end
    eq(true, vim.startswith(fuzzy.backend(), "native"), "precondition: the native backend is the active one")

    local cands = corpus()
    local ctx = fuzzy.prepare(cands)
    eq(true, ctx.native, "precondition: the prepared context really is native")

    local queries = {
        "a",
        "i",
        "fb",
        "fz",
        "init",
        "lua",
        "lvfz",
        "my_n",
        "FB",
        "mn",
        "xy",
        "sc",
        "zzz",
        "",
        "lvimfuzzyinitlua",
    }
    for _, query in ipairs(queries) do
        local native_res, native_n = fuzzy.match(query, ctx)
        local lua_res, lua_n = fallback.match(cands, query, config.max_results or 0)
        eq(lua_n, native_n, ("count differs for %q"):format(query))
        for k = 1, lua_n do
            eq(
                { index = lua_res[k].index, score = lua_res[k].score },
                { index = native_res[k].index, score = native_res[k].score },
                ("result %d differs for %q: %s vs %s"):format(
                    k,
                    query,
                    cands[lua_res[k].index],
                    cands[native_res[k].index]
                )
            )
        end
    end
    fuzzy.free(ctx)
end)

it("the stateless filter agrees with the prepared match", function()
    local cands = corpus()
    local ctx = fuzzy.prepare(cands)
    for _, query in ipairs({ "fb", "init", "my_n", "x" }) do
        local prepared = fuzzy.match(query, ctx)
        local stateless = fuzzy.filter(cands, query)
        eq(#prepared, #stateless, "a one-off rank must not differ from a prepared one")
        for k = 1, #prepared do
            eq(prepared[k], stateless[k])
        end
    end
    fuzzy.free(ctx)
end)

it("a context can grow without re-ranking what was already there", function()
    local first = { "alpha", "beta" }
    local ctx = fuzzy.prepare(first)
    local before = fuzzy.match("a", ctx)
    truthy(#before >= 1)

    fuzzy.append(ctx, { "gamma_a", "delta" })
    local after = fuzzy.match("a", ctx)
    truthy(#after > #before, "the new candidates have to be reachable")

    local names = {}
    for _, r in ipairs(after) do
        names[#names + 1] = ctx.candidates[r.index]
    end
    truthy(vim.tbl_contains(names, "gamma_a"), "an appended candidate was never ranked: " .. vim.inspect(names))
    fuzzy.free(ctx)
end)
