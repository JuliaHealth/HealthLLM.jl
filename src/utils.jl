module Utils

using PromptingTools
using RAGTools

"""
    build_index_rag(cfg, files; embedder_kwargs=())

Build a RAG index from source files using RAGTools.

# Arguments
- `cfg`: A RAGTools indexer configuration (e.g. `RAGTools.SimpleIndexer()`).
- `files`: Vector of file paths or text sources to index.

# Keywords
- `embedder_kwargs=()`: Additional keyword arguments passed to the embedder (e.g.
  `(model="nomic-embed-text",)`).

# Returns
A RAG index object suitable for querying with `generate_funsql_query`.

# Example

```julia
index = build_index_rag(RAGTools.SimpleIndexer(), ["doc1.md", "doc2.md"])
```
"""
function build_index_rag(cfg, files; embedder_kwargs=())
    return RAGTools.build_index(cfg, files; embedder_kwargs=embedder_kwargs)
end

"""
    get_schema(schema_name=nothing, model=nothing)

Return a PromptingTools schema instance inferred from `schema_name` or `model`.

Tries to construct `<Name>Schema()` from PromptingTools when available, falls
back to `PromptingTools.OllamaSchema()`.

# Arguments
- `schema_name::Union{Nothing,String}`: Explicit schema name (e.g. `"Ollama"`, `"HuggingFace"`).
- `model::Union{Nothing,String}`: Model name used for heuristic detection (e.g. `"hf:..."` triggers HuggingFaceSchema).

# Returns
A PromptingTools schema instance.

# Example

```julia
get_schema("Ollama")                    # -> OllamaSchema()
get_schema(nothing, "hf:facebook/opt-350m")  # -> HuggingFaceSchema()
```
"""
function get_schema(schema_name::Union{Nothing,String}=nothing, model::Union{Nothing,String}=nothing)
    if schema_name !== nothing
        ctor = Symbol(string(schema_name) * "Schema")
        if isdefined(PromptingTools, ctor)
            return getfield(PromptingTools, ctor)()
        end
    end

    if model !== nothing
        low = lowercase(model)
        if startswith(low, "hf:") || occursin("huggingface", low) || occursin("hf/", low) || occursin("hf-", low)
            if isdefined(PromptingTools, :HuggingFaceSchema)
                return PromptingTools.HuggingFaceSchema()
            end

            return PromptingTools.OllamaSchema()
        end
    end

    return PromptingTools.OllamaSchema()
end

"""
    HuggingFaceLoadResult

Result of a HuggingFace model load operation.

# Fields
- `path::Union{String,Nothing}`: Local path to the downloaded model snapshot, or `nothing` if not downloaded.
- `info::Any`: Additional metadata from HuggingFaceHub (e.g. model info), or `nothing`.
- `downloaded::Bool`: Whether the model was successfully downloaded.
"""
struct HuggingFaceLoadResult
    path::Union{String,Nothing}
    info::Any
    downloaded::Bool
end

"""
    load_huggingface_model(model; token=nothing)

Download and load a HuggingFace model by name using HuggingFaceHub.jl.

# Arguments
- `model::String`: HuggingFace model identifier (e.g. `"gpt2"`, `"facebook/opt-350m"`).

# Keywords
- `token::Union{Nothing,String}`: HuggingFace API token for private/gated models.

# Returns
- `HuggingFaceLoadResult`: Contains `path`, `info`, and `downloaded` fields.

# Example

```julia
res = load_huggingface_model("gpt2")
if res.downloaded
    println("Model downloaded to \$(res.path)")
end
```
"""
function load_huggingface_model(model::String; token::Union{Nothing,String}=nothing)
    if isdefined(Main, :HuggingFaceHub)
        try
            HF = Main.HuggingFaceHub
            if isdefined(HF, :snapshot)
                if token === nothing
                    p = HF.snapshot(model)
                else
                    p = HF.snapshot(model; token=token)
                end
                return HuggingFaceLoadResult(string(p), nothing, true)
            elseif isdefined(HF, :repo_download)
                if token === nothing
                    p = HF.repo_download(model)
                else
                    p = HF.repo_download(model; token=token)
                end
                return HuggingFaceLoadResult(string(p), nothing, true)
            else
                try
                    info = HF.info(HF.Model, model)
                    if isdefined(HF, :file_download)
                        localdir = HF.file_download(info, ".")
                        return HuggingFaceLoadResult(string(localdir), info, true)
                    else
                        return HuggingFaceLoadResult(nothing, info, false)
                    end
                catch err
                    @warn "HuggingFaceHub helpers present but download failed: $err"
                    return HuggingFaceLoadResult(nothing, nothing, false)
                end
            end
        catch err
            @warn "HuggingFaceHub.jl present but operation failed: $err"
            return HuggingFaceLoadResult(nothing, nothing, false)
        end
    end

    @warn "HuggingFaceHub.jl not available — returning model string. Install HuggingFaceHub.jl for direct downloads."
    return HuggingFaceLoadResult(nothing, model, false)
end

"""
    collect_files_with_extensions(directory, extensions)

Recursively collect file paths under `directory` whose extension is in `extensions`.
Extensions are matched case-insensitively (e.g. `".jl"`).

# Arguments
- `directory::AbstractString`: Root directory to search recursively.
- `extensions::AbstractVector{<:AbstractString}`: File extensions to match (e.g. `[".jl", ".md"]`).

# Returns
- `Vector{String}`: Sorted list of full file paths matching the given extensions.

# Example

```julia
files = collect_files_with_extensions("src", [".jl", ".md"])
```
"""
function collect_files_with_extensions(
    directory::AbstractString,
    extensions::AbstractVector{<:AbstractString}
)
    normalized_extensions = Set(lowercase.(String.(extensions)))
    files = String[]
    for (root, _, file_names) in walkdir(directory)
        for file_name in file_names
            extension = lowercase(splitext(file_name)[2])
            if extension in normalized_extensions
                full_path = joinpath(root, file_name)
                push!(files, full_path)
            end
        end
    end
    return files
end

"""
    write_combined_file(files, output_file)

Concatenate the provided text files into a single output file and include
a `# File: <path>` header before each file body.

# Arguments
- `files::AbstractVector{<:AbstractString}`: List of file paths to concatenate.
- `output_file::AbstractString`: Path to the output file.

# Returns
- `String`: The path to the output file.

# Example

```julia
files = collect_files_with_extensions("data", [".jl", ".md"])
combined = write_combined_file(files, "combined.txt")
```
"""
function write_combined_file(
    files::AbstractVector{<:AbstractString},
    output_file::AbstractString
)
    open(output_file, "w") do io
        for file in files
            println(io, "# File: $file")
            open(file, "r") do f
                for line in eachline(f)
                    println(io, line)
                end
            end
            println(io, "\n")
        end
    end
    return output_file
end

"""
    register_models(model_name, model_embedding; schema_name=nothing)

Register chat and embedding models in PromptingTools and set them as active defaults.

# Arguments
- `model_name::String`: Name of the chat/generation model (e.g. `"llama3.2"`, `"hf:facebook/opt-350m"`).
- `model_embedding::String`: Name of the embedding model (e.g. `"nomic-embed-text"`, `"hf:all-MiniLM-L6-v2"`).

# Keywords
- `schema_name::Union{Nothing,String}`: Optional explicit schema name (e.g. `"Ollama"`, `"HuggingFace"`).
  If `nothing`, the schema is inferred from model names.

# Returns
`nothing`.

# Example

```julia
register_models("llama3.2", "nomic-embed-text")
register_models("hf:facebook/opt-350m", "hf:all-MiniLM-L6-v2"; schema_name="HuggingFace")
```
"""
function register_models(model_name::String, model_embedding::String; schema_name::Union{Nothing,String}=nothing)
    schema_for_generator = get_schema(schema_name, model_name)
    schema_for_embedder = get_schema(schema_name, model_embedding)

    PromptingTools.register_model!(name=model_name, schema=schema_for_generator)
    PromptingTools.register_model!(name=model_embedding, schema=schema_for_embedder)

    PromptingTools.MODEL_CHAT = model_name
    PromptingTools.MODEL_EMBEDDING = model_embedding
    return nothing
end

end
