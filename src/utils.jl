"""
    Utils

Cross-cutting helpers shared by the rest of HealthLLM: model/schema resolution,
HuggingFace model loading, provenance rendering, optional-dependency lookup, and
small file utilities.
"""
module Utils

using PromptingTools
using RAGTools
import HuggingFaceHub
using ..HuggingFace: HuggingFaceOpenAISchema

export build_index_rag, get_schema, register_models,
    HuggingFaceLoadResult, load_huggingface_model,
    require_main_module, render_provenance, check_dims, check_k,
    collect_files_with_extensions, write_combined_file

"""
    check_dims(embeddings, chunks=nothing, expected_dim=nothing) -> (dim, n)

Shape check for a `dim × n` embedding matrix whose columns are chunks — the
convention used throughout embedding, storage, and database code. Verifies the
row count against `expected_dim` and the column count against `length(chunks)`
when either is supplied, then returns `size(embeddings)`.

Throws `DimensionMismatch` on the first mismatch. Every layer that accepts an
embedding matrix alongside its texts routes its shape check through here, so the
same misuse reports the same way everywhere.
"""
function check_dims(embeddings::AbstractMatrix,
    chunks::Union{Nothing,AbstractVector}=nothing,
    expected_dim::Union{Nothing,Integer}=nothing)
    dim, n = size(embeddings)
    if expected_dim !== nothing && dim != expected_dim
        throw(DimensionMismatch(
            "embedding dimension ($dim) does not match expected dimension ($expected_dim)"))
    end
    if chunks !== nothing && length(chunks) != n
        throw(DimensionMismatch(
            "number of chunks ($(length(chunks))) does not match embedding columns ($n)"))
    end
    return (dim, n)
end

"""
    check_k(k) -> Int

Validate a nearest-neighbour count, throwing `ArgumentError` unless `k > 0`.
"""
function check_k(k::Integer)
    k > 0 || throw(ArgumentError("k must be positive, got $k"))
    return Int(k)
end

"""
    build_index_rag(cfg, files; embedder_kwargs=NamedTuple())

Build a RAG index from source files using RAGTools.

# Arguments
- `cfg`: A RAGTools indexer configuration (e.g. `RAGTools.SimpleIndexer()`).
- `files`: Vector of file paths or text sources to index.

# Keywords
- `embedder_kwargs=NamedTuple()`: Additional keyword arguments passed to the embedder (e.g.
  `(model="nomic-embed-text",)`).

# Returns
A RAG index object suitable for querying with `generate_funsql_query`.

# Example

```julia
index = build_index_rag(RAGTools.SimpleIndexer(), ["doc1.md", "doc2.md"])
```
"""
function build_index_rag(cfg, files; embedder_kwargs=NamedTuple())
    return RAGTools.build_index(cfg, files; embedder_kwargs=embedder_kwargs)
end

"""
    require_main_module(name::Symbol, hint::AbstractString) -> Module

Return the module `name` if the driver session has loaded it into `Main`,
otherwise throw an `ArgumentError` carrying `hint`.

Optional backends (FAISS, FunSQL) are deliberately *not* hard dependencies of
HealthLLM: they are heavy, and only some workflows need them. This is the single
place that resolves such a module, so every optional backend fails the same way
with an actionable message instead of a `MethodError` deep inside a call.

# Example

```julia
FunSQL = require_main_module(:FunSQL, "Install FunSQL.jl and run `using FunSQL`.")
```
"""
function require_main_module(name::Symbol, hint::AbstractString)
    isdefined(Main, name) && return getfield(Main, name)
    throw(ArgumentError("$name is not loaded. $hint"))
end

"""
    render_provenance(md::AbstractDict; separator=" › ", maxlen=512) -> String

Render grounding metadata as a single provenance string,
`"<url-or-source> › <heading-or-group>"`, truncated to `maxlen` characters.

Either half may be missing: with no parent heading only the base is returned, and
with no base only the parent. Both the retrieval-index provenance recorded by
`chunk_provenance` and the per-chunk tag rendered into a prompt by
`format_context` go through here, so the two cannot drift apart.

# Example

```julia
render_provenance(Dict{Symbol,Any}(:url => "http://cdm", :heading => "person"))
# "http://cdm › person"
```
"""
function render_provenance(md::AbstractDict; separator::AbstractString=" › ", maxlen::Integer=512)
    base = string(get(md, :url, ""))
    isempty(base) && (base = string(get(md, :source, "")))
    parent = string(get(md, :heading, get(md, :group, "")))
    prov = isempty(parent) ? base :
           isempty(base) ? parent :
           string(base, separator, parent)
    return first(prov, maxlen)
end

"""
    get_schema(provider::Symbol)
    get_schema(schema_name=nothing, model=nothing)

Return a PromptingTools schema instance.

The `Symbol` method is the explicit form: `:ollama` and `:huggingface` map
directly onto their schemas. The two-argument method is the heuristic form used
when only user-supplied strings are available — it tries `<Name>Schema` from
`schema_name`, then sniffs `model` for a HuggingFace marker (`"hf:"`, ...), and
finally falls back to `PromptingTools.OllamaSchema()`.

HuggingFace resolves to [`HuggingFaceOpenAISchema`](@ref), which this package
defines: PromptingTools has no HuggingFace schema of its own.

# Example

```julia
get_schema(:ollama)                          # -> OllamaSchema()
get_schema(:huggingface)                     # -> HuggingFaceOpenAISchema()
get_schema("HuggingFace")                    # -> HuggingFaceOpenAISchema()
get_schema(nothing, "hf:facebook/opt-350m")  # -> HuggingFaceOpenAISchema()
```
"""
function get_schema(provider::Symbol)
    provider === :ollama && return PromptingTools.OllamaSchema()
    provider === :huggingface && return HuggingFaceOpenAISchema()
    throw(ArgumentError("provider must be :ollama or :huggingface, got :$provider"))
end

function get_schema(schema_name::Union{Nothing,String}=nothing, model::Union{Nothing,String}=nothing)
    if schema_name !== nothing
        _looks_like_huggingface(schema_name) && return HuggingFaceOpenAISchema()
        ctor = Symbol(string(schema_name) * "Schema")
        isdefined(PromptingTools, ctor) && return getfield(PromptingTools, ctor)()
    end

    model !== nothing && _looks_like_huggingface(model) && return HuggingFaceOpenAISchema()

    return PromptingTools.OllamaSchema()
end

# Recognises both a schema name ("HuggingFace") and a model reference
# ("hf:org/repo"), hence `name` rather than `model`.
function _looks_like_huggingface(name::AbstractString)
    low = lowercase(name)
    return startswith(low, "hf:") || occursin("huggingface", low) ||
           occursin("hf/", low) || occursin("hf-", low)
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

The download entry point is probed at run time (`snapshot`, then `repo_download`,
then `info`/`file_download`) because HuggingFaceHub.jl has moved it across
versions. Any failure — no network, gated repo, unknown model — is reported as a
warning and returned as a non-downloaded result rather than thrown.

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
    auth = token === nothing ? NamedTuple() : (; token=token)

    for entry in (:snapshot, :repo_download)
        isdefined(HuggingFaceHub, entry) || continue
        try
            path = getproperty(HuggingFaceHub, entry)(model; auth...)
            return HuggingFaceLoadResult(string(path), nothing, true)
        catch err
            @warn "HuggingFaceHub.$entry failed for $model" exception = err
            return HuggingFaceLoadResult(nothing, nothing, false)
        end
    end

    try
        info = HuggingFaceHub.info(HuggingFaceHub.Model, model)
        isdefined(HuggingFaceHub, :file_download) ||
            return HuggingFaceLoadResult(nothing, info, false)
        return HuggingFaceLoadResult(string(HuggingFaceHub.file_download(info, ".")), info, true)
    catch err
        @warn "HuggingFaceHub metadata lookup/download failed for $model" exception = err
        return HuggingFaceLoadResult(nothing, nothing, false)
    end
end

"""
    collect_files_with_extensions(directory, extensions)

Recursively collect file paths under `directory` whose extension is in `extensions`.
Extensions are matched case-insensitively (e.g. `".jl"`).

# Arguments
- `directory::AbstractString`: Root directory to search recursively.
- `extensions::AbstractVector{<:AbstractString}`: File extensions to match (e.g. `[".jl", ".md"]`).

# Returns
- `Vector{String}`: List of full file paths matching the given extensions.

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
                push!(files, joinpath(root, file_name))
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
    PromptingTools.register_model!(name=model_name, schema=get_schema(schema_name, model_name))
    PromptingTools.register_model!(name=model_embedding, schema=get_schema(schema_name, model_embedding))

    PromptingTools.MODEL_CHAT = model_name
    PromptingTools.MODEL_EMBEDDING = model_embedding
    return nothing
end

end
