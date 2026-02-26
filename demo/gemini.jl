using HealthLLM
using RAGTools
using PromptingTools
using GoogleGenAI
using Printf
using LinearAlgebra

const COMBINED_PATH = joinpath(@__DIR__, "JuliaHealthLLM_exp_raw_combined.txt")
const GENAI_MODEL = "gemini-2.0-flash"
const EMBED_MODEL = "gemini-embedding-001"
const GOOGLE_SCHEMA = PromptingTools.GoogleSchema()

get_google_key() = get(ENV, "GOOGLE_API_KEY", "")

using MozillaCACerts_jll
ENV["SSL_CERT_FILE"] = MozillaCACerts_jll.cacert

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

function get_gemini_embeddings(api_key::String, texts::Vector{String}; model::String=EMBED_MODEL)
    embeds = Vector{Vector{Float32}}()
    for text in texts
        result = GoogleGenAI.embed_content(api_key, model, text)
        embedding = Float32.(result.embedding)
        push!(embeds, embedding)
    end
    return embeds
end

function build_index(api_key::String, files::AbstractVector{<:AbstractString}; 
                    chunk_size::Int=1000, overlap::Int=100)
    all_chunks = String[]
    
    for file in files
        text = read(file, String)
        chunks = chunk_text(text; chunk_size=chunk_size, overlap=overlap)
        append!(all_chunks, chunks)
    end
    
    println("Computing embeddings for $(length(all_chunks)) chunks...")
    embeddings = get_gemini_embeddings(api_key, all_chunks)
    
    for i in eachindex(embeddings)
        emb = embeddings[i]
        norm = norm(emb)
        if norm > 0
            embeddings[i] = emb ./ norm
        end
    end
    
    emb_matrix = hcat(embeddings...)
    index = RAGTools.ChunkIndex(
        chunks=all_chunks,
        embeddings=emb_matrix
    )
    return index
end

function retrieve_chunks(index, question::AbstractString; topk::Int=5, api_key::String)
    q_emb = GoogleGenAI.embed_content(api_key, EMBED_MODEL, question)
    query_vec = Float32.(q_emb.embedding)
    query_vec = query_vec ./ norm(query_vec)
    
    scores = index.embeddings' * query_vec
    idxs = sortperm(scores; rev=true)[1:min(topk, length(scores))]
    
    results = RAGTools.RAGResult[]
    for idx in idxs
        if scores[idx] > 0
            push!(results, RAGTools.RAGResult(
                text=index.chunks[idx],
                score=scores[idx],
                source="Chunk$idx",
                metadata=Dict{Symbol, Any}()
            ))
        end
    end
    return results
end

function generate_answer(index, question::AbstractString; api_key::String, topk::Int=5)
    chunks = retrieve_chunks(index, question; topk=topk, api_key=api_key)
    
    context = join([c.text for c in chunks], "\n\n---\n\n")
    
    prompt = """
    Use the following retrieved context to answer the question concisely.
    For any given query write its FunSQL query. Give the funsql query as output.
    Give only the FunSQL query as output nothing else.

    CONTEXT:
    $context

    QUESTION:
    $question

    ANSWER:
    """
    
    response = GoogleGenAI.aigenerate(api_key, GENAI_MODEL, prompt)
    return (answer=response.content, chunks=chunks)
end

function main()
    api_key = get_google_key()
    isempty(api_key) && throw(ArgumentError("GOOGLE_API_KEY must be set in environment"))
    
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

    @printf("Building RAG index from %s ...\n", COMBINED_PATH)
    index = build_index(api_key, [COMBINED_PATH])
    @printf("Index built with %d chunks\n", length(index.chunks))

    println("\n" * "="^50)
    println("HealthLLM.jl Demo - Gemini RAG (Single API Key)")
    println("="^50)
    println("Ask questions about your health data.")
    println("Type an empty line to exit.\n")

    while true
        print("> ")
        q = readline()
        isempty(strip(q)) && break
        
        try
            println("\n--- Retrieving relevant chunks ---")
            chunks = retrieve_chunks(index, q; topk=5, api_key=api_key)
            for (i, chunk) in enumerate(chunks)
                @printf("[%d] Score: %.4f\n", i, chunk.score)
                txt = chunk.text
                println(first(txt, min(300, length(txt))), (length(txt) > 300 ? "..." : ""))
                println()
            end

            println("\n--- Generated FunSQL Query ---")
            result = generate_answer(index, q; api_key=api_key)
            println(result.answer)
            println()
        catch err
            @printf("ERROR: %s\n", err)
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
