module Ingestion

using RAGTools
using PromptingTools

function load_documents(paths::Vector{String})
    docs = Pair{String,String}[]
    for p in paths
        if isfile(p)
            text = read(p, String)
            fname = basename(p)
            push!(docs, fname => text)
        elseif isdir(p)
            for (root, _, files) in walkdir(p)
                for f in files
                    fp = joinpath(root, f)
                    text = read(fp, String)
                    push!(docs, basename(fp) => text)
                end
            end
        end
    end
    docs
end

function build_corpus_index(docs::Vector{Pair{String,String}};
                            chunker::RAGTools.AbstractChunker=RAGTools.TextChunker(),
                            embedder::RAGTools.AbstractEmbedder=RAGTools.BatchEmbedder(),
                            embedder_kwargs::NamedTuple=NamedTuple(),
                            index_id=nothing,
                            verbose::Bool=true)
    sources = first.(docs)
    texts = last.(docs)
    cfg = RAGTools.RAGConfig()
    kwargs = (
        chunker=chunker,
        chunker_kwargs=(sources=sources,),
        embedder=embedder,
        embedder_kwargs=embedder_kwargs,
        verbose=verbose,
    )
    if index_id !== nothing
        kwargs = merge(kwargs, (index_id=index_id,))
    end
    result = RAGTools.build_index(cfg.indexer, texts; kwargs...)
    return result
end

function build_corpus_index_from_files(file_paths::Vector{String}; kwargs...)
    docs = load_documents(file_paths)
    return build_corpus_index(docs; kwargs...)
end

function build_grounding_index(;
                               grounding_dir::Union{String,Nothing}=nothing,
                               embedder_model::Union{String,Nothing}=nothing,
                               chunker::RAGTools.AbstractChunker=RAGTools.TextChunker(),
                               verbose::Bool=true,
                               kwargs...)
    md_dir = grounding_dir !== nothing ? grounding_dir : joinpath(@__DIR__, "..", "docs", "grounding")
    if !isdir(md_dir)
        @warn "Grounding directory not found: $md_dir. Create docs/grounding/ with Markdown reference files."
        md_files = String[]
    else
        md_files = filter(f -> endswith(f, ".md"), readdir(md_dir, join=true))
    end
    funsql_queries_file = joinpath(@__DIR__, "..", "FunSQLQueries", "train.jsonl")
    if isfile(funsql_queries_file)
        push!(md_files, funsql_queries_file)
    end
    isempty(md_files) && error("No grounding documents found. Run with a valid grounding_dir or populate docs/grounding/.")
    if embedder_model !== nothing
        embedder = RAGTools.BatchEmbedder()
        embedder_kwargs = (model=embedder_model,)
    else
        embedder = RAGTools.BatchEmbedder()
        embedder_kwargs = NamedTuple()
    end
    build_corpus_index_from_files(md_files; chunker=chunker, embedder=embedder, embedder_kwargs=embedder_kwargs, verbose=verbose, kwargs...)
end

end
