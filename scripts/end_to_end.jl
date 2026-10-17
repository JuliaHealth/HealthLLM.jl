# End-to-end HealthLLM walkthrough: a natural-language question becomes a FunSQL
# query, a checked SQL statement, and a result table from an OMOP database.
#
#   julia --project=scripts/benchmark scripts/end_to_end.jl \
#       [--db synthea_1M_3YR.duckdb] [--model qwen2.5:7b] [--question "..."]
#
# Needs an OMOP CDM DuckDB file and Ollama serving the chat model and
# nomic-embed-text. The Synthea database is `synthea_1M_3YR.duckdb` in the
# JuliaHealthOrg/JuliaHealthDatasets Hugging Face dataset; docs/src/end-to-end.md
# has the setup. Each step prints what it did, so the output reads as a trace of
# the pipeline.

using HealthLLM
using FunSQL
using DataFrames

include(joinpath(@__DIR__, "benchmark", "omop_duckdb.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_QUESTION = "How many distinct patients had a condition era of essential hypertension " *
                         "(concept 320128) starting in 2020, broken down by gender concept name?"

function arg(name, default)
    i = findfirst(==("--" * name), ARGS)
    return i === nothing ? default : ARGS[i+1]
end

step(n, title) = println("\n", "="^72, "\n", n, ". ", title, "\n", "="^72)

function main()
    db = arg("db", joinpath(ROOT, "synthea_1M_3YR.duckdb"))
    model = arg("model", "qwen2.5:7b")
    embedding = "nomic-embed-text"
    question = arg("question", DEFAULT_QUESTION)

    step(1, "Models and database")
    register_models(model, embedding)
    (; conn, catalog, tables) = open_omop_duckdb(db)
    println("chat model: $model, embedding model: $embedding")
    println("database: $db ($(length(tables)) OMOP tables, read-only)")

    step(2, "Build the retrieval index")
    # Grounding docs (schema, FunSQL reference, translation patterns) plus the
    # verified FunSQL examples. Examples asking this exact question are left out,
    # so the answer is composed rather than looked up.
    tasks = load_tasks(joinpath(ROOT, "data", "benchmark", "funsql_tasks.jsonl"))
    examples = filter(t -> t.question != question, tasks)
    docs = vcat(load_grounding_docs(), examples_document(examples))
    chunks = reduce(vcat, chunk_document.(docs))
    texts = [c.text for c in chunks]
    E = embed(texts, embedding)
    store = LocalVectorStore(size(E, 1))
    add!(store, E, texts)
    println("indexed $(length(texts)) chunks from $(length(docs)) documents")

    step(3, "Retrieve context for the question")
    println("question: ", question, "\n")
    hits = [(; text=chunks[h.index].text, metadata=chunks[h.index].metadata, score=h.score)
            for h in retrieve(store, question, 6; model=embedding)]
    for (i, h) in enumerate(hits)
        println(rpad("[$i]", 5), rpad(round(h.score; digits=3), 7), render_provenance(h.metadata))
    end

    step(4, "Build the grounded prompt")
    prompt = build_prompt(question, hits)
    println("system prompt: $(length(prompt.system)) characters (FUNSQL_SYSTEM_PROMPT)")
    println("user turn: $(length(prompt.user)) characters (retrieved context + question)")

    step(5, "Generate FunSQL")
    gen = generate_funsql(prompt; model=model, verbose=false,
        api_kwargs=(; options=(; temperature=0.0, seed=42)))
    println(gen.funsql)

    step(6, "Check it against the schema")
    check = sanity_check_funsql(gen; catalog=catalog)
    println(check)
    if !check.ok
        println("\nThe query did not pass the check, so it is not executed:\n", check.error)
        return
    end
    println("\nrendered SQL:\n", check.sql)

    step(7, "Run it")
    result = execute_with_timeout(conn, check.sql; timeout=120)
    show(stdout, MIME("text/plain"), result; allrows=true)
    println()
end

main()
