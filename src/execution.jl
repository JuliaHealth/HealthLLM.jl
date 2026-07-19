"""
    Execution

Turn a grounded prompt into a FunSQL query and check that the query the model
produced is actually usable. This closes the loop opened by [`Prompt`](@ref):

```
build_prompt ─▶ generate_funsql ─▶ FunSQLGeneration ─▶ sanity_check_funsql ─▶ FunSQLCheck
                (call the model)    (extracted code)     (parse / build / render / run)
```

Two responsibilities:

1. **Generation** — [`generate_funsql`](@ref) sends the grounded prompt to a chat
   model and extracts the FunSQL code block from the reply
   ([`extract_funsql`](@ref)).
2. **Execution-based sanity check** — [`sanity_check_funsql`](@ref) confirms the
   generated code *parses* as Julia, *builds* into a `FunSQL.SQLNode`, and
   *renders* to SQL — optionally against a schema catalog, which is what catches a
   query that references a table or column the OMOP CDM does not have. With a
   connection it can also *run* the SQL.

## Optional dependency

Building and rendering FunSQL needs the `FunSQL` package loaded in `Main` (it is a
test/driver-side dependency, not a hard dependency of HealthLLM, mirroring how
[`FaissVectorStore`](@ref) treats `Faiss`). The parse stage works without it; the
later stages raise a clear error until `using FunSQL` has run.

!!! warning "Evaluating model output"
    [`sanity_check_funsql`](@ref) `eval`s the generated code to build the query
    object — the same approach the FunSQL test suite uses. Evaluated code runs with
    full privileges, so only check output from a model you trust, and never point
    the `conn` execution path at a production database.
"""
module Execution

import PromptingTools
using ..Prompt: build_prompt, PromptTemplate, DEFAULT_FUNSQL_TEMPLATE

export extract_funsql, generate_funsql, FunSQLGeneration,
    sanity_check_funsql, FunSQLCheck

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

"""
    FunSQLGeneration

Result of [`generate_funsql`](@ref).

# Fields
- `funsql::String`: The FunSQL code extracted from the model's reply — the code
  block only, ready to hand to [`sanity_check_funsql`](@ref).
- `answer::String`: The model's full reply (code block plus any explanation).
"""
struct FunSQLGeneration
    funsql::String
    answer::String
end

const _JULIA_FENCE = r"```(?:julia|jl)[ \t]*\r?\n(.*?)```"s
const _ANY_FENCE = r"```[A-Za-z0-9]*[ \t]*\r?\n(.*?)```"s

"""
    extract_funsql(text::AbstractString) -> String

Pull the FunSQL code out of a model reply. Prefers the first fenced block tagged
` ```julia ` / ` ```jl `, falls back to the first fenced block of any language,
and finally to the whole stripped `text` when no fence is present. The returned
code is stripped of surrounding whitespace.
"""
function extract_funsql(text::AbstractString)
    m = match(_JULIA_FENCE, text)
    m === nothing && (m = match(_ANY_FENCE, text))
    body = m === nothing ? text : m.captures[1]
    return String(strip(body))
end

# Turn a build_prompt result (or a plain string) into what aigenerate accepts:
# a system/user conversation when the grounding split is available, else a string.
_to_conversation(p::AbstractString) = String(p)
function _to_conversation(p)
    if p isa NamedTuple && haskey(p, :system) && haskey(p, :user)
        return [PromptingTools.SystemMessage(p.system), PromptingTools.UserMessage(p.user)]
    elseif p isa NamedTuple && haskey(p, :prompt)
        return String(p.prompt)
    end
    return string(p)
end

_message_content(msg) = hasproperty(msg, :content) ? String(msg.content) : string(msg)

function _default_llm_generator(prompt; model, schema=nothing, kwargs...)
    conv = _to_conversation(prompt)
    msg = schema === nothing ?
          PromptingTools.aigenerate(conv; model=model, kwargs...) :
          PromptingTools.aigenerate(schema, conv; model=model, kwargs...)
    return _message_content(msg)
end

"""
    generate_funsql(prompt; model=PromptingTools.MODEL_CHAT, schema=nothing,
                    generator=<aigenerate>, kwargs...) -> FunSQLGeneration

Generate a FunSQL query from a grounded `prompt` and extract the code block from
the reply. `prompt` is typically the NamedTuple returned by [`build_prompt`](@ref)
— its `system`/`user` split is sent as a proper chat conversation so the grounding
rules land in the system role — but a plain prompt string works too.

`model` is the chat model name (defaults to the active `PromptingTools.MODEL_CHAT`).
`schema` is an optional `PromptingTools` schema; when given it is passed to
`aigenerate` as the first argument. Extra `kwargs` are forwarded to the generator.

`generator` is injectable for testing: it is called as
`generator(prompt; model, schema, kwargs...)` and must return the reply text. The
default calls `PromptingTools.aigenerate`.

# Example

```julia
hits = retrieve(store, question, 6)
p    = build_prompt(question, hits)
gen  = generate_funsql(p; model = "llama3.2")
gen.funsql     # the extracted FunSQL code
```
"""
function generate_funsql(prompt;
    model::AbstractString=PromptingTools.MODEL_CHAT,
    schema=nothing,
    generator=_default_llm_generator,
    kwargs...)
    text = generator(prompt; model=model, schema=schema, kwargs...)
    return FunSQLGeneration(extract_funsql(text), String(text))
end

# ---------------------------------------------------------------------------
# Execution-based sanity check
# ---------------------------------------------------------------------------

"""
    FunSQLCheck

Outcome of [`sanity_check_funsql`](@ref). Records how far the generated query got
through the parse → build → render → run pipeline and why it stopped.

# Fields
- `ok::Bool`: Reached the deepest requested stage with no error.
- `stage::Symbol`: Deepest stage reached — `:parsed`, `:built`, `:rendered`, or
  `:executed`. On failure this is the last stage that *succeeded*.
- `parsed::Bool`: The code parsed as Julia.
- `built::Bool`: Evaluating it produced a `FunSQL.SQLNode`.
- `rendered::Bool`: The node rendered to SQL (against `catalog`, when supplied).
- `executed::Bool`: The SQL ran via `executor` (only when one was given).
- `sql::Union{String,Nothing}`: The rendered SQL, when rendering succeeded.
- `error::Union{String,Nothing}`: Message from the failing stage, or `nothing`.
"""
struct FunSQLCheck
    ok::Bool
    stage::Symbol
    parsed::Bool
    built::Bool
    rendered::Bool
    executed::Bool
    sql::Union{String,Nothing}
    error::Union{String,Nothing}
end

function Base.show(io::IO, c::FunSQLCheck)
    status = c.ok ? "ok" : "failed"
    print(io, "FunSQLCheck($status @ :$(c.stage)")
    c.error === nothing || print(io, ", error=", repr(c.error))
    print(io, ")")
end

_funsql_module() = isdefined(Main, :FunSQL) ? Main.FunSQL :
    throw(ArgumentError(
        "FunSQL is required to build/render a query. Install FunSQL.jl and run " *
        "`using FunSQL` before calling sanity_check_funsql with build/render stages."))

# FunSQL does not export its node constructors (`From`, `Get`, `Agg`, ...), so
# generated code that uses the bare names does not resolve under a plain
# `using FunSQL`. We eval it in a dedicated sandbox module that binds those names
# (and pulls in the `@funsql`/`funsql_*` DSL), so callers need not import anything
# into `Main`. Built lazily on first use because FunSQL is a Main-only dependency.
const _FUNSQL_NODE_NAMES = (
    :From, :Select, :Where, :Join, :LeftJoin, :Group, :Order, :Limit, :Get, :Agg,
    :Fun, :As, :Define, :Bind, :Partition, :Append, :With, :Iterate, :Lit, :Var,
    :Sort, :Asc, :Desc, :Over, :Highlight,
)
const _EVAL_MOD = Ref{Module}()

function _funsql_eval_module()
    isassigned(_EVAL_MOD) && return _EVAL_MOD[]
    FunSQL = _funsql_module()
    m = Module(:FunSQLSandbox)
    Core.eval(m, :(const FunSQL = $FunSQL))
    Core.eval(m, :(using FunSQL))                       # @funsql + funsql_* exports
    for name in _FUNSQL_NODE_NAMES                      # the capitalized node API
        isdefined(FunSQL, name) &&
            Core.eval(m, :(const $name = $(getproperty(FunSQL, name))))
    end
    _EVAL_MOD[] = m
    return m
end

"""
    sanity_check_funsql(code::AbstractString; catalog=nothing, dialect=:duckdb,
                        mod=nothing, executor=nothing) -> FunSQLCheck
    sanity_check_funsql(gen::FunSQLGeneration; kwargs...) -> FunSQLCheck

Check that generated FunSQL `code` is usable, without trusting the model's word
for it. Runs as far through the pipeline as the inputs allow and reports the
result as a [`FunSQLCheck`](@ref):

1. **parse** — `code` parses as Julia. Needs nothing beyond Julia itself.
2. **build** — evaluating it yields a `FunSQL.SQLNode`. Requires `FunSQL` loaded in
   `Main` (see the module note). By default the code is evaluated in a built-in
   sandbox module that already binds the FunSQL node constructors (`From`, `Get`,
   `Agg`, ...), so bare-name generations resolve without importing anything into
   `Main`. Pass `mod` to evaluate in your own module instead.
3. **render** — the node serialises to SQL. With `catalog` (a `FunSQL.SQLCatalog`)
   the query is resolved **against that schema**, so a reference to a table or
   column the schema does not contain fails here — this is the check that catches a
   hallucinated OMOP name that slipped past the prompt. Without a catalog the
   render is structural, using `dialect` (a dialect name like `:duckdb` or a
   `FunSQL.SQLDialect`).
4. **run** — when `executor` is supplied, the rendered SQL string is passed to it
   (e.g. `sql -> DuckDB.execute(conn, sql)`), confirming the database accepts it.

The check never throws for a bad query: any stage failure is captured in the
returned `error` with `ok = false`. It only throws for a genuine misuse (build/
render requested without FunSQL available).

# Examples

```julia
using FunSQL
catalog = FunSQL.SQLCatalog(
    FunSQL.SQLTable(:person, columns = [:person_id, :year_of_birth]),
    dialect = :duckdb,
)

sanity_check_funsql("From(:person) |> Group() |> Select(:n => Agg.count())";
                    catalog = catalog)
# FunSQLCheck(ok @ :rendered)

sanity_check_funsql("From(:person) |> Select(Get.made_up_column)"; catalog = catalog)
# FunSQLCheck(failed @ :built)   -> error names the unresolved column
```
"""
function sanity_check_funsql(code::AbstractString; catalog=nothing, dialect=:duckdb,
    mod::Union{Module,Nothing}=nothing, executor=nothing)
    parsed = built = rendered = executed = false
    sql = nothing

    # 1. parse — wrap in a block so multi-line generations parse as one unit.
    expr = try
        Meta.parse(string("begin\n", code, "\nend"))
    catch err
        return FunSQLCheck(false, :none, false, false, false, false, nothing, _errmsg(err))
    end
    parsed = true

    # 2. build — eval to a FunSQL.SQLNode.
    FunSQL = _funsql_module()
    eval_mod = mod === nothing ? _funsql_eval_module() : mod
    node = try
        Core.eval(eval_mod, expr)
    catch err
        return FunSQLCheck(false, :parsed, parsed, false, false, false, nothing, _errmsg(err))
    end
    node isa FunSQL.SQLNode ||
        return FunSQLCheck(false, :parsed, parsed, false, false, false, nothing,
            "generated code evaluated to a $(typeof(node)), not a FunSQL.SQLNode")
    built = true

    # 3. render — against the schema catalog when given, else structurally.
    sql = try
        rendered_sql = catalog === nothing ?
                       FunSQL.render(node; dialect=dialect) :
                       FunSQL.render(catalog, node)
        string(rendered_sql)
    catch err
        return FunSQLCheck(false, :built, parsed, built, false, false, nothing, _errmsg(err))
    end
    rendered = true

    # 4. run — only when an executor is supplied.
    if executor !== nothing
        try
            executor(sql)
            executed = true
        catch err
            return FunSQLCheck(false, :rendered, parsed, built, rendered, false, sql, _errmsg(err))
        end
    end

    stage = executed ? :executed : :rendered
    return FunSQLCheck(true, stage, parsed, built, rendered, executed, sql, nothing)
end

sanity_check_funsql(gen::FunSQLGeneration; kwargs...) =
    sanity_check_funsql(gen.funsql; kwargs...)

_errmsg(err) = first(sprint(showerror, err), 500)

end
