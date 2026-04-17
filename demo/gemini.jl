using HealthLLM
using Printf
using ConfigEnv

dotenv(joinpath(@__DIR__, "..", ".env"))

function main()
    println("\n" * "="^50)
    println("HealthLLM.jl Demo")
    println("Generation: Gemini")
    println("="^50)
    
    embedder = SimpleEmbedder()
    
    data_dir = get(
        ENV,
        "HEALTHLLM_EXP_RAW_ROOT",
        joinpath(dirname(@__DIR__), "JuliaHealthLLM", "data", "exp_raw")
    )
    
    combined_path = joinpath(@__DIR__, "OHDSI_FunSQL_combined.txt")
    if !isfile(combined_path)
        isdir(data_dir) || throw(ArgumentError("Data directory does not exist: $data_dir"))
        @printf("Preparing data from %s ...\n", data_dir)
        prepare_data(data_dir, output=combined_path)
    else
        @printf("Using existing combined file: %s\n", combined_path)
    end
    
    @printf("\nBuilding RAG index from %s ...\n", combined_path)
    index = build_index(embedder, [combined_path])
    @printf("Index built with %d chunks\n", length(index.chunks))
    
    provider = GeminiProvider()
    
    println("\nAsk questions about your health data.")
    println("Type an empty line to exit.\n")
    
    while true
        print("> ")
        q = strip(readline())
        isempty(q) && break
        q = String(q)
        
        try
            println("\n--- Retrieving relevant chunks ---")
            chunks = retrieve_chunks(index, q; topk=5, embedder=embedder)
            for (i, chunk) in enumerate(chunks)
                @printf("[%d] Score: %.4f\n", i, chunk.score)
                txt = chunk.text
                println(first(txt, min(300, length(txt))), (length(txt) > 300 ? "..." : ""))
                println()
            end

            println("\n--- Generated FunSQL Query ---")
            result = generate_answer(index, q; embedder=embedder, provider=provider)
            println(result.answer)
            println()
        catch err
            @printf("ERROR: %s\n", err)
            showerror(stderr, stacktrace(catch_backtrace()))
        end
    end
    println("bye")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

if !isinteractive()
    main()
end