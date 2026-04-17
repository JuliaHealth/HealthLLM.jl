module RAG

using ..HealthLLM: collect_files_with_extensions, write_combined_file, ModelProvider

export SimpleEmbedder, prepare_data, chunk_text, build_index, ChunkResult, retrieve_chunks, generate_answer

struct SimpleEmbedder
    vocab::Dict{String, Int}
    dimension::Int
end

function SimpleEmbedder(dimension::Int=1024)
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

function prepare_data(data_dir::AbstractString; output::AbstractString)
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

function generate_answer(index, question::AbstractString; topk::Int=5, embedder::SimpleEmbedder, provider::ModelProvider)
    chunks = retrieve_chunks(index, question; topk=topk, embedder=embedder)
    
    context = join([c.text for c in chunks], "\n\n---\n\n")
    
    prompt = """
    Use the following retrieved context to answer the question.
    Write a Julia expression using FunSQL.jl to generate the appropriate SQL query.
    Give only the Julia FunSQL code as output, nothing else.

    CONTEXT:
    $context

    QUESTION:
    $question

    ANSWER:
    """
    
    answer = generate(provider, prompt)
    return (answer=answer, chunks=chunks)
end

end