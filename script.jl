using HealthLLM
using RAGTools
using PromptingTools

mktempdir() do dir
    write(joinpath(dir, "doc1.md"), "# Hypertension\nTreatment includes lifestyle changes and medication.")
    write(joinpath(dir, "doc2.md"), "# Diabetes\nManagement involves diet and insulin.")
    write(joinpath(dir, "utils.jl"), "function foo() end")

    # 1. Collect files
    files = collect_files_with_extensions(dir, [".md", ".jl"])
    println("1. Collected files: ", files)

    # 2. Register models (Ollama or HuggingFace)
    register_models("llama3.2", "nomic-embed-text")
    println("2. Models registered: chat=$(PromptingTools.MODEL_CHAT), embedding=$(PromptingTools.MODEL_EMBEDDING)")

    # 3. Build RAG index
    try
        index = build_index_rag(RAGTools.SimpleIndexer(), files)
        println("3. Index built successfully")

        # 4. Validate embeddings
        chunks = ["chunk $i" for i in 1:length(files)]
        embeddings = rand(384, length(files))
        validate_embeddings_inputs(embeddings, chunks, 384)
        println("4. Embeddings validated (384-dim, $(length(chunks)) chunks)")

        # 5. Store embeddings (requires PostgreSQL with pgvector)
        println("5. store_embeddings_pgvector skipped — needs PostgreSQL with pgvector")

        # 6. Query
        answer = generate_funsql_query(
            index, "nomic-embed-text", "llama3.2",
            "Context: {input_query}. Answer concisely.",
            "What is hypertension?"
        )
        println("6. Answer: ", answer)
    catch e
        println("3-6. Stopped at build_index_rag (expected — needs Ollama running): ", e)
    end
end
