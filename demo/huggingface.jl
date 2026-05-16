using HealthLLM
using RAGTools
using PromptingTools
using Printf
using LinearAlgebra
using ConfigEnv

dotenv(joinpath(@__DIR__, "..", ".env"))

const HF_MODEL = "Qwen/Qwen3.5-0.8B"
const HF_API_KEY = get(ENV, "HF_TOKEN", get(ENV, "HUGGINGFACE_TOKEN", ""))
const HF_ENDPOINT = get(ENV, "HF_API_ENDPOINT", "https://router.huggingface.co/chat/completions")
const EMBEDDING_DIM = 1024
const OLLAMA_MODEL = get(ENV, "OLLAMA_MODEL", "qwen2.5:1.5b")
const OLLAMA_URL = get(ENV, "OLLAMA_URL", "http://localhost:11434")

struct SimpleEmbedder
    vocab::Dict{String, Int}
    dimension::Int
end

function SimpleEmbedder(dimension::Int=EMBEDDING_DIM)
    println("Initializing simple embedder (dimension: $dimension)")
    return SimpleEmbedder(Dict{String, Int}(), dimension)
end

function tokenize(text::String)
    lowercase.(split(replace(text, r"[^\w\s]" => " ")))
end

function get_embedding(embedder::SimpleEmbedder, text::String)
    words = tokenize(text)
    vector = zeros(Float32, embedder.dimension)
    
    for word in words
        hash_idx = hash(word) % embedder.dimension + 1
        vector[hash_idx] += 1.0f0
    end
    
    norm_vec = norm(vector)
    if norm_vec > 0
        vector ./= norm_vec
    end
    
    return vector
end

function get_embeddings(embedder::SimpleEmbedder, texts::Vector{String})
    return [get_embedding(embedder, text) for text in texts]
end

normalize!(embeddings::Vector{Vector{Float32}}) = foreach(embeddings) do emb
    n = norm(emb)
    n > 0 && (emb ./= n)
end

function prepare_data(data_dir::AbstractString; output::AbstractString=COMBINED_PATH)
    files = collect_files_with_extensions(data_dir, [".md", ".jl", ".txt", ".csv"])
    write_combined_file(files, output)
    return output
end

function chunk_text(text::String; chunk_size::Int=1000, overlap::Int=100)
    chunks = String[]
    chars = collect(text)
    pos = 1
    while pos <= length(chars)
        endpos = min(pos + chunk_size - 1, length(chars))
        push!(chunks, strip(join(chars[pos:endpos])))
        pos = endpos + 1
    end
    return chunks
end

function build_index(embedder::SimpleEmbedder, files::AbstractVector{<:AbstractString}; 
                   chunk_size::Int=1000, overlap::Int=100)
    all_chunks = String[]
    
    for file in files
        text = read(file, String)
        chunks = chunk_text(text; chunk_size=chunk_size, overlap=overlap)
        append!(all_chunks, chunks)
    end
    
    println("Computing embeddings for $(length(all_chunks)) chunks...")
    embeddings = get_embeddings(embedder, all_chunks)
    
    emb_matrix = hcat(embeddings...)
    index = RAGTools.ChunkIndex(chunks=all_chunks, embeddings=emb_matrix, sources=all_chunks)
    return index
end

struct ChunkResult
    text::String
    score::Float32
    source::String
end

ChunkResult(; text, score, source) = ChunkResult(text, score, source)

function retrieve_chunks(index, question::AbstractString; topk::Int=5, embedder::SimpleEmbedder)
    query_emb = get_embedding(embedder, question)
    
    scores = index.embeddings' * query_emb
    idxs = sortperm(scores; rev=true)[1:min(topk, length(scores))]
    
    results = ChunkResult[]
    for idx in idxs
        if scores[idx] > 0
            push!(results, ChunkResult(;
                text=index.chunks[idx],
                score=scores[idx],
                source="Chunk$idx"
            ))
        end
    end
    return results
end

function generate_answer(index, question::AbstractString; topk::Int=5, embedder::SimpleEmbedder, ollama_llm=nothing)
    chunks = retrieve_chunks(index, question; topk=topk, embedder=embedder)
    
    context = join([c.text for c in chunks], "\n\n---\n\n")
    
    prompt = """
    Use the following retrieved context to answer the question.
    For any given query write its FunSQL query. Give the funsql query as output.
    Give only the FunSQL query as output nothing else.

    CONTEXT:
    $context

    QUESTION:
    $question

    ANSWER:
    """
    
    if ollama_llm !== nothing
        println("Generating with Ollama ($(ollama_llm.model_name))...")
        answer = generate(ollama_llm, prompt)
    else
        response = aigenerate(
            PromptingTools.CustomOpenAISchema(),
            prompt;
            model=HF_MODEL,
            api_key=HF_API_KEY,
            api_kwargs=(; url=HF_ENDPOINT)
        )
        answer = response.content
    end
    return (answer=answer, chunks=chunks)
end

function main()
    println("Using HuggingFace model: $HF_MODEL")
    println("Endpoint: $HF_ENDPOINT")
    
    ollama_llm = nothing
    println("\nTrying to load Ollama model: $OLLAMA_MODEL")
    println("Ollama URL: $OLLAMA_URL")
    
    if ollama_is_running(OLLAMA_URL)
        try
            ollama_llm = load_ollama_model(OLLAMA_MODEL; base_url=OLLAMA_URL)
            set_generation_config!(ollama_llm; temperature=0.3f0, num_predict=512)
            println("Ollama model loaded successfully!")
        catch e
            println("Warning: Could not load Ollama model: $e")
            println("Falling back to API-based inference...")
        end
    else
        println("Ollama is not running at $OLLAMA_URL")
        println("Start it with: ollama serve")
        println("Or install Ollama from: https://ollama.com/download")
        println("\nFalling back to API-based inference...")
    end
    
    data_dir = get(
        ENV,
        "HEALTHLLM_EXP_RAW_ROOT",
        joinpath(dirname(@__DIR__), "JuliaHealthLLM", "data", "exp_raw")
    )
    
    if !isfile(COMBINED_PATH)
        isdir(data_dir) || throw(ArgumentError("Data directory does not exist: $data_dir"))
        @printf("Preparing data from %s ...\n", data_dir)
        prepare_data(data_dir)
    else
        @printf("Using existing combined file: %s\n", COMBINED_PATH)
    end
    
    println("\n" * "="^50)
    println("HealthLLM.jl Demo")
    println("Embedding: Local Hash Embeddings (Open Source)")
    println("Generation: HuggingFace ($HF_MODEL)")
    println("="^50)
    
    embedder = SimpleEmbedder()
    
    @printf("\nBuilding RAG index from %s ...\n", COMBINED_PATH)
    index = build_index(embedder, [COMBINED_PATH])
    @printf("Index built with %d chunks\n", length(index.chunks))
    
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
            result = generate_answer(index, q; embedder=embedder, ollama_llm=ollama_llm)
            println(result.answer)
            println()
        catch err
            @printf("ERROR: %s\n", err)
            showerror(stderr, stacktrace(catch_backtrace()))
        end
    end
    println("bye")
end

const COMBINED_PATH = joinpath(@__DIR__, "JuliaHealthLLM_exp_raw_combined.txt")

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

if !isinteractive()
    main()
end
