module Utils

using PromptingTools
using RAGTools

function build_index_rag(cfg, files; embedder_kwargs=())
    return RAGTools.build_index(cfg, files; embedder_kwargs=embedder_kwargs)
end

"""
Return a PromptingTools schema instance inferred from `schema_name` or `model`.
Tries to construct `<Name>Schema()` from PromptingTools when available, falls
back to `PromptingTools.OllamaSchema()`.
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

struct HuggingFaceLoadResult
    path::Union{String,Nothing}
    info::Any
    downloaded::Bool
end

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
