-- lvim-fuzzy.fallback: the pure-Lua matcher — a BYTE-EXACT mirror of the native
-- Rust scorer (native/src/lib.rs). It runs when the .so is absent (no Rust
-- toolchain) so the plugin always works; because it implements the IDENTICAL
-- affine-gap recurrence, bonus constants, m==1 closed form and deterministic
-- total order (score desc → shorter candidate → index asc), its ranking is
-- indistinguishable from the native backend — a consumer never sees different
-- results depending on whether the binary is present. Parity is asserted at
-- runtime by health.lua (diffs the two backends when both are available).
--
---@module "lvim-fuzzy.fallback"

local byte = string.byte

local M = {}

-- Scoring constants — MUST stay identical to native/src/lib.rs.
local BONUS_FIRST = 18
local BONUS_BOUNDARY = 16
local BONUS_CAMEL = 14
local BONUS_CONSEC = 16
local GAP_LEADING = -1
local GAP_INNER = -2
local GAP_TRAILING = -1
local NEG = -1073741824 -- i32::MIN/2, matching the native −∞ sentinel
local MATCH_FLOOR = -16000 -- scores above this are real matches (native parity)

-- boundary bytes: / _ - . <space> :
---@type table<integer, boolean>
local BOUNDARY = { [47] = true, [95] = true, [45] = true, [46] = true, [32] = true, [58] = true }

--- Lowercase one ASCII byte (A–Z → a–z; others unchanged).
---@param b integer
---@return integer
local function lower(b)
    if b >= 65 and b <= 90 then
        return b + 32
    end
    return b
end

--- m == 1 closed form: best positional bonus of any matching byte minus the
--- (L-1) gap toll (native `score_m1`). Returns NEG when the char is absent.
---@param cand string
---@param want integer  lowercased needle byte
---@return integer
local function score_m1(cand, want)
    local l = #cand
    local toll = l - 1
    if lower(byte(cand, 1)) == want then
        return BONUS_FIRST - toll
    end
    local best = NEG
    local prev = byte(cand, 1)
    for j = 2, l do
        local cb = byte(cand, j)
        if lower(cb) == want then
            if BOUNDARY[prev] then
                return BONUS_BOUNDARY - toll
            elseif prev >= 97 and prev <= 122 and cb >= 65 and cb <= 90 then
                if BONUS_CAMEL > best then
                    best = BONUS_CAMEL
                end
            elseif best < 0 then
                best = 0
            end
        end
        prev = cb
    end
    if best == NEG then
        return NEG
    end
    return best - toll
end

-- Reused DP columns (two of D and M), swapped by reference each byte — mirrors
-- the native column-major single-pass affine-gap DP with O(m) state.
local dp, mp, dc, mc = {}, {}, {}, {}

--- Affine-gap alignment score of `lc_needle` (m ≥ 2, precondition: subsequence
--- of `cand`) against `cand` (native `score`). Byte-exact with the Rust path.
---@param cand string
---@param lc_needle integer[]  lowercased needle bytes, 1-indexed
---@param m integer
---@return integer
local function score(cand, lc_needle, m)
    for i = 1, m do
        dp[i] = NEG
        mp[i] = NEG
    end
    local l = #cand
    for j = 1, l do
        local cb = byte(cand, j)
        local lc = lower(cb)
        local bonus
        if j == 1 then
            bonus = BONUS_FIRST
        else
            local p = byte(cand, j - 1)
            if BOUNDARY[p] then
                bonus = BONUS_BOUNDARY
            elseif p >= 97 and p <= 122 and cb >= 65 and cb <= 90 then
                bonus = BONUS_CAMEL
            else
                bonus = 0
            end
        end
        local lead = (j - 1) * GAP_LEADING -- native j is 0-based
        for i = 1, m do
            local dval
            if lc == lc_needle[i] then
                local fresh = (i == 1) and lead or mp[i - 1]
                local consec = (i == 1) and NEG or dp[i - 1]
                local a = fresh + bonus
                local b = consec + bonus + BONUS_CONSEC
                dval = (a > b) and a or b
            else
                dval = NEG
            end
            local gap = (i == m) and GAP_TRAILING or GAP_INNER
            dc[i] = dval
            local m_gap = mp[i] + gap
            mc[i] = (dval > m_gap) and dval or m_gap
        end
        dp, dc = dc, dp
        mp, mc = mc, mp
    end
    return mp[m] -- after the final swap the last column lives in `mp`
end

--- Is `lc_needle` a case-folded subsequence of `cand`? (native `is_subseq`)
---@param cand string
---@param lc_needle integer[]
---@param m integer
---@return boolean
local function is_subseq(cand, lc_needle, m)
    local ni = 1
    local want = lc_needle[1]
    for j = 1, #cand do
        if lower(byte(cand, j)) == want then
            ni = ni + 1
            if ni > m then
                return true
            end
            want = lc_needle[ni]
        end
    end
    return false
end

--- Rank `candidates` against `query`. Deterministic total order (score desc →
--- shorter candidate → 1-based index asc), capped to `max_results` (0 = no cap).
--- Byte-exact with the native backend.
---@param candidates string[]
---@param query string
---@param max_results integer
---@return { index: integer, score: integer }[] results
---@return integer count
function M.match(candidates, query, max_results)
    local n = #candidates
    local cap = (max_results and max_results > 0) and math.min(n, max_results) or n

    -- empty query → identity: first `cap` candidates in source order, score 0
    if query == "" then
        local out = {}
        for i = 1, cap do
            out[i] = { index = i, score = 0 }
        end
        return out, cap
    end

    -- lowercase the needle once (1-indexed byte array)
    local m = #query
    local lc_needle = {}
    for i = 1, m do
        lc_needle[i] = lower(byte(query, i))
    end

    local hits = {}
    if m == 1 then
        local want = lc_needle[1]
        for i = 1, n do
            local cand = candidates[i]
            if #cand > 0 then
                local sc = score_m1(cand, want)
                if sc > MATCH_FLOOR then
                    hits[#hits + 1] = { index = i, score = sc, len = #cand }
                end
            end
        end
    else
        for i = 1, n do
            local cand = candidates[i]
            if #cand >= m and is_subseq(cand, lc_needle, m) then
                local sc = score(cand, lc_needle, m)
                if sc > MATCH_FLOOR then
                    hits[#hits + 1] = { index = i, score = sc, len = #cand }
                end
            end
        end
    end

    -- deterministic total order: score desc, shorter candidate, index asc
    table.sort(hits, function(a, b)
        if a.score ~= b.score then
            return a.score > b.score
        end
        if a.len ~= b.len then
            return a.len < b.len
        end
        return a.index < b.index
    end)

    local cnt = math.min(#hits, cap)
    local out = {}
    for k = 1, cnt do
        out[k] = { index = hits[k].index, score = hits[k].score }
    end
    return out, cnt
end

return M
