# Zero-shot vs RAG benchmark for natural-language -> FunSQL generation.
#
#   julia --project=scripts/benchmark scripts/benchmark/run_benchmark.jl \
#       --db synthea_1M_3YR.duckdb --model qwen2.5:7b
#
# Every task in data/benchmark/funsql_tasks.jsonl is answered by the same model
# twice: once with no context, and once with chunks retrieved from the grounding
# docs (data/grounding/). Each generated query is rendered against the real
# schema, executed, and its result compared with the gold SQL's. Results go to
# benchmark_results/<timestamp>/ as results.json (every task) and report.md.
#
# Options (defaults in brackets):
#   --db PATH             OMOP DuckDB file                      [synthea_1M_3YR.duckdb]
#   --tasks PATH          benchmark tasks                       [data/benchmark/funsql_tasks.jsonl]
#   --model NAME          Ollama chat model                     [qwen2.5:7b]
#   --embedding NAME      Ollama embedding model                [nomic-embed-text]
#   --k N                 chunks retrieved per question         [6]
#   --modes LIST          zero_shot,rag                         [zero_shot,rag]
#   --tiers LIST          simple,compositional,schema_aware     [all]
#   --limit N             evaluate only the first N tasks       [all]
#   --test-fraction X     fraction of tasks evaluated. The rest become retrievable
#                         examples; their verified FunSQL is indexed   [1.0]
#   --timeout S           per-query execution timeout, seconds  [60]
#   --out DIR             output directory                      [benchmark_results]

using HealthLLM
using FunSQL
using JSON3
using Dates
import PromptingTools

include(joinpath(@__DIR__, "omop_duckdb.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function parse_args(args)
    opts = Dict{String,String}(
        "db" => joinpath(ROOT, "synthea_1M_3YR.duckdb"),
        "tasks" => joinpath(ROOT, "data", "benchmark", "funsql_tasks.jsonl"),
        "model" => "qwen2.5:7b", "embedding" => "nomic-embed-text", "k" => "6",
        "modes" => "zero_shot,rag", "tiers" => join(String.(TIERS), ","), "limit" => "0",
        "test-fraction" => "1.0", "timeout" => "60", "out" => joinpath(ROOT, "benchmark_results"))
    i = 1
    while i <= length(args)
        key = lstrip(args[i], '-')
        haskey(opts, key) && i < length(args) || error("unknown or incomplete option: $(args[i])")
        opts[key] = args[i+1]
        i += 2
    end
    return opts
end

# Index the grounding docs, plus the verified FunSQL of any tasks not under test,
# in an in-memory vector store. The retriever returns chunk text, provenance and
# score, which is what `build_prompt` renders and `TaskResult` records.
function build_retriever(examples, embedding, k)
    docs = load_grounding_docs()
    isempty(examples) || push!(docs, examples_document(examples))
    chunks = reduce(vcat, chunk_document.(docs))
    texts = [c.text for c in chunks]
    @info "Embedding $(length(texts)) chunks from $(length(docs)) documents with $embedding"
    E = embed(texts, embedding)
    store = LocalVectorStore(size(E, 1))
    add!(store, E, texts)
    retriever = question -> [(; text=chunks[h.index].text, metadata=chunks[h.index].metadata,
        score=h.score) for h in retrieve(store, question, k; model=embedding)]
    return retriever, length(texts)
end

function main(args)
    opts = parse_args(args)
    modes = Tuple(Symbol.(split(opts["modes"], ',')))
    tiers = Tuple(Symbol.(split(opts["tiers"], ',')))
    k = parse(Int, opts["k"])

    register_models(opts["model"], opts["embedding"])
    (; conn, catalog) = open_omop_duckdb(opts["db"])
    executor = duckdb_executor(conn; timeout=parse(Float64, opts["timeout"]))

    tasks = load_tasks(opts["tasks"]; tiers=tiers)
    (; examples, test) = split_tasks(tasks; test_fraction=parse(Float64, opts["test-fraction"]))
    limit = parse(Int, opts["limit"])
    limit > 0 && (test = test[1:min(limit, end)])
    @info "Evaluating $(length(test)) tasks ($(join(("$t=$(count(x -> x.tier === t, test))" for t in TIERS), ", "))) in modes $(modes)"

    retriever, n_chunks = :rag in modes ? build_retriever(examples, opts["embedding"], k) : (nothing, 0)

    started = now()
    results = run_benchmark(test; modes=modes, model=opts["model"], executor=executor,
        catalog=catalog, retriever=retriever, verbose=false,
        api_kwargs=(; options=(; temperature=0.0, seed=42)))

    meta = (; model=opts["model"], embedding=opts["embedding"], k=k, tasks=basename(opts["tasks"]),
        evaluated=length(test), indexed_chunks=n_chunks,
        indexed_examples=count(t -> !isempty(t.gold_funsql), examples),
        temperature=0.0, started=string(started), minutes=round((now() - started).value / 60_000; digits=1),
        healthllm=string(pkgversion(HealthLLM)), funsql=string(pkgversion(FunSQL)))

    dir = joinpath(opts["out"], Dates.format(started, "yyyy-mm-dd_HHMMSS"))
    mkpath(dir)
    write_results(joinpath(dir, "results.json"), results; meta=meta)
    report = markdown_report(results; meta=meta)
    write(joinpath(dir, "report.md"), report)
    println("\n", report, "\nSaved to ", dir)
end

main(ARGS)
