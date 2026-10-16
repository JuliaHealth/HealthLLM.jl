"""
    Evaluation

A benchmark harness for natural-language → FunSQL generation that measures
whether retrieval grounding actually helps. Every task is run twice through the
same model — once **zero-shot** (no retrieved context) and once **RAG** (context
from a vector store) — and each output is scored by *executing* it, not by
string similarity to a reference.

```
BenchmarkTask ─▶ prompt (zero-shot | RAG) ─▶ generate_funsql ─▶ sanity_check_funsql
                                                                   │
                       gold SQL ─▶ executor ─▶ ResultSet ◀─ executor ◀┘
                                                   │
                                          results_match ─▶ TaskResult
```

## Scoring

Each [`TaskResult`](@ref) records the deepest stage the generated query reached
and why it stopped (`outcome`):

| `outcome`        | meaning                                                  |
|------------------|----------------------------------------------------------|
| `:correct`       | executed, and its result matches the gold SQL's result   |
| `:mismatch`      | executed, but returned a different result                |
| `:execute_error` | rendered to SQL, but the database rejected it            |
| `:render_error`  | built a query that does not resolve against the schema (e.g. a hallucinated column) |
| `:build_error`   | parsed, but evaluating it did not yield a `FunSQL.SQLNode` |
| `:parse_error`   | not valid Julia                                          |
| `:generation_error` | the model call itself failed                          |
| `:gold_error`    | the task's own gold SQL failed — the task is excluded from accuracy |

**Execution accuracy** — the headline metric — is the fraction of scorable tasks
whose generated query returns the same result as the gold SQL (see
[`results_match`](@ref) for what "the same" means). [`compare_modes`](@ref)
pairs the two runs task-by-task and reports the discordant counts plus an exact
McNemar test, so a RAG gain can be told apart from noise.

## Leakage

The benchmark questions come from the same FunSQL example set the RAG index can
draw on. Indexing a task's own gold answer and then asking that question measures
lookup, not grounding. [`split_tasks`](@ref) holds the evaluated tasks out, and
[`examples_document`](@ref) turns only the *remaining* tasks into retrievable
examples.

## Dependencies

The harness itself depends on nothing beyond HealthLLM: the database is reached
through an injected `executor(sql) -> ResultSet`, the model through `generate_funsql`'s
injectable `generator`, and retrieval through an injected `retriever(question) -> hits`.
Building and rendering the generated FunSQL requires `FunSQL` loaded in `Main`,
as for [`sanity_check_funsql`](@ref). `scripts/benchmark/run_benchmark.jl` wires all
of this to DuckDB and Ollama.
"""
module Evaluation

using JSON3
using ..Prompt: build_prompt, PromptTemplate, DEFAULT_FUNSQL_TEMPLATE
using ..Execution: generate_funsql, sanity_check_funsql, FunSQLGeneration
using ..Ingestion: SourceDocument

export BenchmarkTask, TIERS, classify_tier, load_tasks, save_tasks, split_tasks,
    examples_document, ResultSet, results_match, ZERO_SHOT_SYSTEM_PROMPT,
    zero_shot_prompt, TaskResult, OUTCOMES, run_task, run_benchmark,
    summarize, compare_modes, mcnemar_exact, write_results, markdown_report

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

"""
    TIERS

Benchmark difficulty tiers, in increasing order of what the model must know:

- `:simple` — one OMOP table: filter, aggregate, group, order.
- `:compositional` — several clauses composed: joins, subqueries, CTEs, set or
  window operations.
- `:schema_aware` — needs the OMOP *vocabulary* (`concept`, `concept_ancestor`,
  ...) to resolve a clinical idea to concept ids, i.e. domain-model knowledge
  rather than query syntax alone.
"""
const TIERS = (:simple, :compositional, :schema_aware)

"""
    BenchmarkTask

One natural-language → FunSQL benchmark task.

# Fields
- `id::String`: Stable identifier, used to pair zero-shot and RAG runs.
- `question::String`: The natural-language analytical question.
- `group::String`: The OMOP area the task belongs to (e.g. `"drug_exposure"`).
- `tier::Symbol`: One of [`TIERS`](@ref).
- `gold_sql::String`: Reference SQL whose result defines the correct answer.
- `gold_funsql::String`: Reference FunSQL (used as a retrievable example, never
  as the scoring reference).
- `description::String`: Longer task description (may be empty).
"""
struct BenchmarkTask
    id::String
    question::String
    group::String
    tier::Symbol
    gold_sql::String
    gold_funsql::String
    description::String
end

const _VOCAB_TABLE = r"\b(?:\w+\.)?(concept|concept_ancestor|concept_relationship|concept_synonym|vocabulary|domain|concept_class|relationship|source_to_concept_map|drug_strength)\b"i
const _COMPOSITIONAL = r"\bJOIN\b|\bWITH\b|\(\s*SELECT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|\bOVER\s*\("i

"""
    classify_tier(sql::AbstractString) -> Symbol

Assign a [`TIERS`](@ref) tier from a task's gold SQL: `:schema_aware` when it
reads an OMOP vocabulary table, else `:compositional` when it joins, nests a
subquery, defines a CTE, or uses a set/window operation, else `:simple`.

A heuristic over the reference query, so it labels what the *answer* needs rather
than how the question is phrased. Override it by giving a task an explicit
`"tier"` field in its JSONL record.
"""
function classify_tier(sql::AbstractString)
    occursin(_VOCAB_TABLE, sql) && return :schema_aware
    occursin(_COMPOSITIONAL, sql) && return :compositional
    return :simple
end

_str(rec, key, default="") = haskey(rec, key) && rec[key] !== nothing ? strip(string(rec[key])) : default

"""
    load_tasks(path; tiers=TIERS) -> Vector{BenchmarkTask}

Load benchmark tasks from a JSON Lines file in the FunSQL example format
(`query`, `group`, `description`, `sql_query`, `response`, plus optional `id` and
`tier`). Records missing a question or gold SQL are skipped. A record without an
`id` gets `"task-<line>"`; one without a `tier` is classified with
[`classify_tier`](@ref). Only tasks whose tier is in `tiers` are returned.
"""
function load_tasks(path::AbstractString; tiers=TIERS)
    isfile(path) || throw(ArgumentError("No such benchmark file: $path"))
    tasks = BenchmarkTask[]
    for (n, line) in enumerate(eachline(path))
        isempty(strip(line)) && continue
        rec = JSON3.read(line)
        question = _str(rec, :query)
        gold_sql = _str(rec, :sql_query)
        (isempty(question) || isempty(gold_sql)) && continue
        tier_str = _str(rec, :tier)
        tier = isempty(tier_str) ? classify_tier(gold_sql) : Symbol(tier_str)
        tier in TIERS || throw(ArgumentError("Unknown tier $(repr(tier)) on line $n of $path"))
        tier in tiers || continue
        push!(tasks, BenchmarkTask(_str(rec, :id, "task-$n"), question,
            _str(rec, :group, "unknown"), tier, gold_sql, _str(rec, :response),
            _str(rec, :description)))
    end
    ids = [t.id for t in tasks]
    length(unique(ids)) == length(ids) || throw(ArgumentError("Duplicate task ids in $path"))
    return tasks
end

_record(t::BenchmarkTask) = (; id=t.id, query=t.question, group=t.group, tier=String(t.tier),
    description=t.description, sql_query=t.gold_sql, response=t.gold_funsql)

"""
    save_tasks(path, tasks) -> path

Write `tasks` as JSON Lines in the format [`load_tasks`](@ref) reads, with
explicit `id` and `tier` fields so a saved benchmark reloads identically.
"""
function save_tasks(path::AbstractString, tasks::AbstractVector{BenchmarkTask})
    open(path, "w") do io
        for t in tasks
            JSON3.write(io, _record(t))
            println(io)
        end
    end
    return path
end

"""
    split_tasks(tasks; test_fraction=0.5, seed=0) -> (; examples, test)

Split `tasks` into a held-out `test` set to evaluate on and an `examples` set the
RAG index may retrieve from. The split is stratified by tier so each tier is
represented in both halves.

Tasks are ordered by a fixed hash of `seed` and their `id` (FNV-1a, not
`Base.hash`), so the split is identical across Julia versions and platforms and
does not change when unrelated tasks are added or removed.
"""
function split_tasks(tasks::AbstractVector{BenchmarkTask}; test_fraction::Real=0.5, seed::Integer=0)
    0 < test_fraction <= 1 || throw(ArgumentError("`test_fraction` must be in (0, 1]."))
    examples = BenchmarkTask[]
    test = BenchmarkTask[]
    for tier in TIERS
        group = sort(filter(t -> t.tier === tier, tasks); by=t -> _fnv1a(string(seed, ':', t.id)))
        n_test = round(Int, test_fraction * length(group))
        append!(test, group[1:n_test])
        append!(examples, group[n_test+1:end])
    end
    return (; examples, test)
end

function _fnv1a(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end

"""
    examples_document(tasks; source="FunSQL-examples") -> SourceDocument

Turn the `tasks` that have a reference FunSQL into a single JSONL [`SourceDocument`](@ref HealthLLM.Ingestion.SourceDocument)
that [`chunk_document`](@ref HealthLLM.Ingestion.chunk_document) splits one
example per chunk ([`RecordChunk`](@ref HealthLLM.Ingestion.RecordChunk)). Pass only the
`examples` half of [`split_tasks`](@ref) so no evaluated task can retrieve its
own answer.
"""
function examples_document(tasks::AbstractVector{BenchmarkTask}; source::AbstractString="FunSQL-examples")
    content = join((JSON3.write(_record(t)) for t in tasks if !isempty(t.gold_funsql)), '\n')
    return SourceDocument(source, "benchmark:examples", "FunSQL examples", content)
end

# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------

"""
    ResultSet(columns, rows)

A query result in a backend-neutral form, so the harness does not depend on any
particular database or table package.

# Fields
- `columns::Vector{String}`: Column names, in order.
- `rows::Vector{Vector{Any}}`: One vector of values per row, aligned with `columns`.

A DataFrame converts as
`ResultSet(names(df), [collect(Any, r) for r in eachrow(df)])`.
"""
struct ResultSet
    columns::Vector{String}
    rows::Vector{Vector{Any}}
end

ResultSet(columns::AbstractVector, rows::AbstractVector) =
    ResultSet(String.(string.(columns)), [collect(Any, r) for r in rows])

Base.size(r::ResultSet) = (length(r.rows), length(r.columns))

_isnull(v) = v === missing || v === nothing
_isnum(v) = v isa Real && !(v isa Bool)

function _values_equal(a, b; atol, rtol)
    _isnull(a) && _isnull(b) && return true
    (_isnull(a) || _isnull(b)) && return false
    _isnum(a) && _isnum(b) && return isapprox(Float64(a), Float64(b); atol=atol, rtol=rtol)
    return isequal(a, b) || string(a) == string(b)
end

# Sort key: numbers rounded so rows that differ only by float noise sort together.
_sortkey(v) = _isnull(v) ? (0, "") : _isnum(v) ? (1, string(round(Float64(v); sigdigits=8))) : (2, string(v))

function _rows_equal(g::Vector{Vector{Any}}, p::Vector{Vector{Any}}; ordered, atol, rtol)
    length(g) == length(p) || return false
    if !ordered
        g = sort(g; by=r -> map(_sortkey, r))
        p = sort(p; by=r -> map(_sortkey, r))
    end
    for (rg, rp) in zip(g, p)
        all(_values_equal(a, b; atol=atol, rtol=rtol) for (a, b) in zip(rg, rp)) || return false
    end
    return true
end

"""
    results_match(gold::ResultSet, pred::ResultSet; ordered=false, atol=1e-6, rtol=1e-4) -> Bool

Execution-accuracy comparison: does `pred` return the same answer as `gold`?

- **Columns** are compared by content, not by name — generated queries rarely
  reproduce the reference's aliases. Both results must have the same number of
  columns. When the column names match as a set (case-insensitively) they are
  aligned by name; otherwise every column ordering of `pred` is tried (up to
  six columns), so `Select(:n, :gender)` matches `Select(:gender, :n)`.
- **Rows** are compared as a multiset unless `ordered=true`: SQL row order is
  only defined under `ORDER BY`, and the questions rarely require one.
- **Values**: numbers compare with `isapprox(; atol, rtol)` (so `COUNT` as
  `Int64` matches `Float64`, and `AVG` tolerates rounding), `missing`/`nothing`
  equal each other, and anything else compares by value or, failing that, by its
  string form (so a `Date` matches its ISO string).
"""
function results_match(gold::ResultSet, pred::ResultSet; ordered::Bool=false,
    atol::Real=1e-6, rtol::Real=1e-4)
    ncol = length(gold.columns)
    length(pred.columns) == ncol || return false
    length(gold.rows) == length(pred.rows) || return false
    ncol == 0 && return true

    gnames = lowercase.(gold.columns)
    pnames = lowercase.(pred.columns)
    perms = if allunique(gnames) && Set(gnames) == Set(pnames)
        [[findfirst(==(n), pnames) for n in gnames]]
    elseif ncol <= 6
        _permutations(collect(1:ncol))
    else
        [collect(1:ncol)]
    end

    for perm in perms
        reordered = [r[perm] for r in pred.rows]
        _rows_equal(gold.rows, reordered; ordered=ordered, atol=atol, rtol=rtol) && return true
    end
    return false
end

# Identity first, so the common already-aligned case is tried before the rest.
function _permutations(v::Vector{Int})
    length(v) <= 1 && return [v]
    out = Vector{Vector{Int}}()
    for i in eachindex(v)
        rest = vcat(v[1:i-1], v[i+1:end])
        for p in _permutations(rest)
            push!(out, vcat(v[i], p))
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

"""
    ZERO_SHOT_SYSTEM_PROMPT

System message for the zero-shot baseline. It asks for the same thing as
[`FUNSQL_SYSTEM_PROMPT`](@ref HealthLLM.Prompt.FUNSQL_SYSTEM_PROMPT) — one FunSQL query over
the OMOP CDM in a fenced `julia` block — but drops the "use only the retrieved
context" grounding rules, which would otherwise tell the model to refuse when
there is no context at all. The difference between the two runs is then the
retrieved context, not a stricter instruction.
"""
const ZERO_SHOT_SYSTEM_PROMPT = """
You are an expert data analyst for the OHDSI OMOP Common Data Model (CDM), \
writing analytical queries with the Julia package FunSQL.jl.

Your task: translate the user's natural-language analytical question into a \
correct FunSQL query against the OMOP CDM, using your knowledge of the OMOP CDM \
tables and columns and of FunSQL.jl.

Output:
- Return the query as ONE fenced code block tagged `julia`, followed by a one- or \
  two-sentence explanation of the tables/columns you used.
- The code block must be a single FunSQL expression (e.g. built with `From`, \
  `Where`, `Join`, `Group`, `Select`, `Agg`, `Fun`, `|>`) — not SQL text.
"""

"""
    zero_shot_prompt(question) -> (; system, user, prompt)

The no-retrieval baseline prompt, in the same `(; system, user, prompt)` shape
[`build_prompt`](@ref HealthLLM.Prompt.build_prompt) returns, so both modes go
through [`generate_funsql`](@ref HealthLLM.Execution.generate_funsql) identically.
"""
function zero_shot_prompt(question::AbstractString)
    q = strip(question)
    isempty(q) && throw(ArgumentError("`question` is empty; nothing to ask."))
    user = string("# Analytical question\n\n", q, "\n\n# FunSQL query")
    return (; system=ZERO_SHOT_SYSTEM_PROMPT, user=user,
        prompt=string(ZERO_SHOT_SYSTEM_PROMPT, "\n\n", user))
end

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

"""
    OUTCOMES

Every value [`TaskResult`](@ref)`.outcome` can take, from best to worst. See the
module docstring for their meaning.
"""
const OUTCOMES = (:correct, :mismatch, :execute_error, :render_error, :build_error,
    :parse_error, :generation_error, :gold_error)

"""
    TaskResult

The outcome of running one [`BenchmarkTask`](@ref) in one mode.

# Fields
- `task_id::String`, `question::String`, `group::String`, `tier::Symbol`: Copied from the task.
- `mode::Symbol`: `:zero_shot` or `:rag`.
- `outcome::Symbol`: One of [`OUTCOMES`](@ref).
- `funsql::String`: The FunSQL code extracted from the model's reply.
- `answer::String`: The model's full reply.
- `sql::Union{String,Nothing}`: The SQL the generated query rendered to.
- `error::Union{String,Nothing}`: Message from the failing stage.
- `context::Vector{String}`: Provenance (or a text excerpt) of each retrieved
  chunk, in rank order. Empty for zero-shot.
- `scores::Vector{Union{Float64,Nothing}}`: Retrieval score of each chunk, or
  `nothing` where the retriever gives none.
- `seconds::Float64`: Wall time for retrieval, generation, and checking.
"""
struct TaskResult
    task_id::String
    question::String
    group::String
    tier::Symbol
    mode::Symbol
    outcome::Symbol
    funsql::String
    answer::String
    sql::Union{String,Nothing}
    error::Union{String,Nothing}
    context::Vector{String}
    scores::Vector{Union{Float64,Nothing}}
    seconds::Float64
end

Base.show(io::IO, r::TaskResult) =
    print(io, "TaskResult(", r.task_id, ", :", r.mode, " => :", r.outcome, ")")

_errmsg(err) = first(sprint(showerror, err), 500)

function _describe_hit(h)
    if hasproperty(h, :metadata) && getproperty(h, :metadata) isa AbstractDict
        md = getproperty(h, :metadata)
        label = string(get(md, :source, ""), " › ", get(md, :heading, get(md, :group, "")))
        strip(label, [' ', '›']) == "" || return label
    end
    text = h isa AbstractString ? h :
           hasproperty(h, :chunk) ? getproperty(h, :chunk) :
           hasproperty(h, :text) ? getproperty(h, :text) : string(h)
    return first(replace(String(text), r"\s+" => " "), 80)
end

_hit_score(h) = hasproperty(h, :score) && getproperty(h, :score) isa Real ?
                Float64(getproperty(h, :score)) : nothing

_stage_outcome(check) = check.stage === :none ? :parse_error :
                        check.stage === :parsed ? :build_error : :render_error

"""
    run_task(task, mode; model, executor, catalog, retriever=nothing,
             template=DEFAULT_FUNSQL_TEMPLATE, gold=nothing, generator=nothing,
             match_kwargs=(;), kwargs...) -> TaskResult

Run one task in `mode` (`:zero_shot` or `:rag`) and score it.

1. Build the prompt: [`zero_shot_prompt`](@ref) or, for `:rag`,
   `build_prompt(question, retriever(question); template)`.
2. Generate with [`generate_funsql`](@ref HealthLLM.Execution.generate_funsql)
   (`model`, `generator`, and `kwargs` are forwarded to it).
3. Parse, build, and render the result against `catalog` (a `FunSQL.SQLCatalog`
   of the target database) with [`sanity_check_funsql`](@ref HealthLLM.Execution.sanity_check_funsql).
4. Run the SQL with `executor(sql) -> ResultSet` and compare it to the gold
   result with [`results_match`](@ref).

`gold` is the task's gold [`ResultSet`](@ref), or an exception if the gold SQL
failed; when `nothing` it is computed with `executor`. [`run_benchmark`](@ref)
computes it once per task and shares it between modes.
"""
function run_task(task::BenchmarkTask, mode::Symbol;
    model::AbstractString, executor, catalog, retriever=nothing,
    template::PromptTemplate=DEFAULT_FUNSQL_TEMPLATE, gold=nothing, generator=nothing,
    match_kwargs=NamedTuple(), kwargs...)
    mode in (:zero_shot, :rag) || throw(ArgumentError("`mode` must be :zero_shot or :rag, got $(repr(mode))."))
    mode === :rag && retriever === nothing &&
        throw(ArgumentError("`:rag` mode needs a `retriever(question) -> hits`."))

    t0 = time()
    if gold === nothing
        gold = try
            executor(task.gold_sql)
        catch err
            err
        end
    end

    context = String[]
    scores = Union{Float64,Nothing}[]
    result(outcome; funsql="", answer="", sql=nothing, error=nothing) =
        TaskResult(task.id, task.question, task.group, task.tier, mode, outcome,
            funsql, answer, sql, error, context, scores, time() - t0)

    gold isa ResultSet || return result(:gold_error; error="gold SQL failed: " *
                                                           (gold isa Exception ? _errmsg(gold) : string(gold)))

    gen = try
        prompt = if mode === :zero_shot
            zero_shot_prompt(task.question)
        else
            hits = collect(retriever(task.question))
            append!(context, _describe_hit.(hits))
            append!(scores, _hit_score.(hits))
            build_prompt(task.question, hits; template=template)
        end
        generator === nothing ?
            generate_funsql(prompt; model=model, kwargs...) :
            generate_funsql(prompt; model=model, generator=generator, kwargs...)
    catch err
        return result(:generation_error; error=_errmsg(err))
    end

    check = sanity_check_funsql(gen; catalog=catalog)
    check.ok || return result(_stage_outcome(check); funsql=gen.funsql, answer=gen.answer,
        error=check.error)

    pred = try
        executor(check.sql)
    catch err
        return result(:execute_error; funsql=gen.funsql, answer=gen.answer, sql=check.sql,
            error=_errmsg(err))
    end

    matched = results_match(gold, pred; match_kwargs...)
    return result(matched ? :correct : :mismatch; funsql=gen.funsql, answer=gen.answer,
        sql=check.sql,
        error=matched ? nothing : "result differs from gold: got $(size(pred)) rows×cols, expected $(size(gold))")
end

"""
    run_benchmark(tasks; modes=(:zero_shot, :rag), progress=true, kwargs...) -> Vector{TaskResult}

Run every task in every mode with [`run_task`](@ref) (`kwargs` are forwarded to
it — `model`, `executor`, `catalog`, and, for `:rag`, `retriever` are required).
Each task's gold SQL is executed once and its result shared by both modes, so the
modes are compared on identical references. Results are ordered by task, then mode.
"""
function run_benchmark(tasks::AbstractVector{BenchmarkTask}; modes=(:zero_shot, :rag),
    executor, progress::Bool=true, kwargs...)
    results = TaskResult[]
    for (i, task) in enumerate(tasks)
        gold = try
            executor(task.gold_sql)
        catch err
            err
        end
        for mode in modes
            r = run_task(task, mode; executor=executor, gold=gold, kwargs...)
            push!(results, r)
            progress && @info "[$i/$(length(tasks))] $(task.id) ($(task.tier)) $(mode): $(r.outcome)"
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_rate(n, d) = d == 0 ? 0.0 : round(n / d; digits=4)

function _stats(rs::AbstractVector{TaskResult})
    scorable = filter(r -> r.outcome !== :gold_error, rs)
    n = length(scorable)
    count_of(o) = count(r -> r.outcome === o, scorable)
    rendered = count(r -> r.outcome in (:correct, :mismatch, :execute_error), scorable)
    executed = count(r -> r.outcome in (:correct, :mismatch), scorable)
    built = rendered + count_of(:render_error)
    parsed = built + count_of(:build_error)
    return (; tasks=n,
        correct=count_of(:correct),
        execution_accuracy=_rate(count_of(:correct), n),
        execution_success=_rate(executed, n),
        render_success=_rate(rendered, n),
        build_success=_rate(built, n),
        parse_success=_rate(parsed, n),
        outcomes=Dict(String(o) => count_of(o) for o in OUTCOMES if o !== :gold_error),
        mean_seconds=n == 0 ? 0.0 : round(sum(r.seconds for r in scorable) / n; digits=2))
end

"""
    summarize(results) -> Dict{Symbol,Any}

Per-mode metrics over `results`. For each mode: the `overall` stats — tasks
scored, execution accuracy (correct ÷ scorable), the success rate of each
pipeline stage (parse → build → render → execute), the outcome histogram, and
mean seconds per task — and the same broken down `by_tier` and `by_group`.
Tasks whose gold SQL failed are reported as `excluded` and left out of every rate.
"""
function summarize(results::AbstractVector{TaskResult})
    out = Dict{Symbol,Any}()
    for mode in unique(r.mode for r in results)
        rs = filter(r -> r.mode === mode, results)
        by_tier = Dict(String(t) => _stats(filter(r -> r.tier === t, rs))
                       for t in TIERS if any(r -> r.tier === t, rs))
        by_group = Dict(g => _stats(filter(r -> r.group == g, rs)) for g in unique(r.group for r in rs))
        out[mode] = (; overall=_stats(rs), by_tier, by_group,
            excluded=count(r -> r.outcome === :gold_error, rs))
    end
    return out
end

"""
    mcnemar_exact(b, c) -> Float64

Two-sided exact McNemar p-value for paired binary outcomes, from the discordant
counts `b` (only the first condition correct) and `c` (only the second correct):
the binomial probability of a split at least as uneven as `min(b, c)` out of
`b + c` under no difference. Concordant pairs carry no information and are not
inputs. Returns `1.0` when there are no discordant pairs.
"""
function mcnemar_exact(b::Integer, c::Integer)
    n = b + c
    n == 0 && return 1.0
    k = min(b, c)
    tail = sum(binomial(big(n), i) for i in 0:k) / big(2)^n
    return Float64(min(1, 2 * tail))
end

"""
    compare_modes(results; baseline=:zero_shot, treatment=:rag) -> NamedTuple

Pair `baseline` and `treatment` results by task and compare their correctness.
Tasks excluded in either mode (`:gold_error`) are dropped. Returns the paired
task count, both accuracies and their difference, the 2×2 agreement counts
(`both_correct`, `treatment_only`, `baseline_only`, `neither`), the exact
McNemar p-value on the discordant pairs, and the ids of the tasks each mode alone
got right — the starting point for error analysis.
"""
function compare_modes(results::AbstractVector{TaskResult};
    baseline::Symbol=:zero_shot, treatment::Symbol=:rag)
    base = Dict(r.task_id => r for r in results if r.mode === baseline)
    treat = Dict(r.task_id => r for r in results if r.mode === treatment)
    ids = sort!([id for id in keys(base) if haskey(treat, id) &&
                 base[id].outcome !== :gold_error && treat[id].outcome !== :gold_error])
    ok(r) = r.outcome === :correct
    both = count(id -> ok(base[id]) && ok(treat[id]), ids)
    t_only = filter(id -> !ok(base[id]) && ok(treat[id]), ids)
    b_only = filter(id -> ok(base[id]) && !ok(treat[id]), ids)
    neither = count(id -> !ok(base[id]) && !ok(treat[id]), ids)
    n = length(ids)
    b_acc = _rate(both + length(b_only), n)
    t_acc = _rate(both + length(t_only), n)
    return (; paired=n, baseline, treatment,
        baseline_accuracy=b_acc, treatment_accuracy=t_acc,
        delta=round(t_acc - b_acc; digits=4),
        both_correct=both, treatment_only=length(t_only), baseline_only=length(b_only),
        neither, p_value=round(mcnemar_exact(length(b_only), length(t_only)); digits=4),
        treatment_only_ids=t_only, baseline_only_ids=b_only)
end

"""
    write_results(path, results; meta=(;)) -> path

Save a benchmark run as JSON: `meta` (model names, dataset, seed, ...), the
[`summarize`](@ref) metrics, the [`compare_modes`](@ref) comparison when both
modes ran, and every [`TaskResult`](@ref) so a run can be re-analysed without
re-querying the model.
"""
function write_results(path::AbstractString, results::AbstractVector{TaskResult}; meta=NamedTuple())
    modes = Set(r.mode for r in results)
    payload = Dict{Symbol,Any}(
        :meta => meta,
        :summary => summarize(results),
        :results => results)
    (:zero_shot in modes && :rag in modes) && (payload[:comparison] = compare_modes(results))
    open(io -> JSON3.pretty(io, payload), path, "w")
    return path
end

_pct(x) = string(round(100x; digits=1), "%")

"""
    markdown_report(results; meta=(;)) -> String

Render a benchmark run as a Markdown report: the headline comparison, per-mode
stage success rates, accuracy by tier, the error-outcome breakdown, and the
tasks each mode alone solved. Intended to be pasted into the evaluation write-up.
"""
function markdown_report(results::AbstractVector{TaskResult}; meta=NamedTuple())
    s = summarize(results)
    modes = [m for m in (:zero_shot, :rag) if haskey(s, m)]
    io = IOBuffer()
    println(io, "# FunSQL generation benchmark\n")
    for (k, v) in pairs(meta)
        println(io, "- **", k, "**: ", v)
    end
    isempty(meta) || println(io)

    if length(modes) == 2
        c = compare_modes(results)
        println(io, "## Zero-shot vs RAG\n")
        println(io, "| | Zero-shot | RAG |\n|---|---|---|")
        println(io, "| Execution accuracy | ", _pct(c.baseline_accuracy), " | ", _pct(c.treatment_accuracy), " |")
        println(io, "\nPaired over ", c.paired, " tasks: RAG ", c.delta >= 0 ? "+" : "", round(100c.delta; digits=1),
            " points. Both correct: ", c.both_correct, ", RAG only: ", c.treatment_only,
            ", zero-shot only: ", c.baseline_only, ", neither: ", c.neither,
            ". Exact McNemar p = ", c.p_value, ".\n")
    end

    println(io, "## Pipeline stage success\n")
    println(io, "| Stage | ", join(String.(modes), " | "), " |")
    println(io, "|---|", repeat("---|", length(modes)))
    for (label, f) in (("Parsed", :parse_success), ("Built", :build_success),
        ("Rendered (schema-valid)", :render_success), ("Executed", :execution_success),
        ("Correct result", :execution_accuracy))
        println(io, "| ", label, " | ", join((_pct(getproperty(s[m].overall, f)) for m in modes), " | "), " |")
    end

    println(io, "\n## Accuracy by tier\n")
    println(io, "| Tier | Tasks | ", join(String.(modes), " | "), " |")
    println(io, "|---|---|", repeat("---|", length(modes)))
    for t in TIERS
        tk = String(t)
        haskey(s[first(modes)].by_tier, tk) || continue
        n = s[first(modes)].by_tier[tk].tasks
        println(io, "| ", tk, " | ", n, " | ",
            join((_pct(s[m].by_tier[tk].execution_accuracy) for m in modes), " | "), " |")
    end

    println(io, "\n## Outcomes\n")
    println(io, "| Outcome | ", join(String.(modes), " | "), " |")
    println(io, "|---|", repeat("---|", length(modes)))
    for o in OUTCOMES
        o === :gold_error && continue
        println(io, "| ", o, " | ", join((s[m].overall.outcomes[String(o)] for m in modes), " | "), " |")
    end
    excluded = s[first(modes)].excluded
    excluded > 0 && println(io, "\n", excluded, " task(s) excluded because their gold SQL failed.")

    if length(modes) == 2
        c = compare_modes(results)
        byid = Dict((r.task_id, r.mode) => r for r in results)
        for (title, ids) in (("Solved only with RAG", c.treatment_only_ids),
            ("Solved only zero-shot", c.baseline_only_ids))
            isempty(ids) && continue
            println(io, "\n## ", title, "\n")
            for id in ids
                println(io, "- `", id, "`: ", byid[(id, :rag)].question)
            end
        end
    end
    return String(take!(io))
end

end
