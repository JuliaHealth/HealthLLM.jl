module Utils

using PromptingTools

"""
Return a PromptingTools schema instance inferred from `schema_name` or `model`.
Tries to construct `<Name>Schema()` from PromptingTools when available, falls
back to `PromptingTools.OllamaSchema()`.
"""
function get_schema(schema_name::Union{Nothing,String}=nothing, model::Union{Nothing,String}=nothing)
    # Prefer an explicit schema name when provided
    if schema_name !== nothing
        ctor = Symbol(string(schema_name) * "Schema")
        if isdefined(PromptingTools, ctor)
            return getfield(PromptingTools, ctor)()
        end
    end

    # Heuristic: if the model string looks like a huggingface model, try HuggingFaceSchema
    if model !== nothing
        low = lowercase(model)
        if startswith(low, "hf:") || occursin("huggingface", low) || occursin("hf/", low) || occursin("hf-", low)
            for schema_name in [:HuggingFaceSchema, :HuggingFaceManagedSchema]
                if isdefined(PromptingTools, schema_name)
                    return getfield(PromptingTools, schema_name)()
                end
            end
            return PromptingTools.OllamaSchema()
        end
    end

    # Default fallback
    return PromptingTools.OllamaSchema()
end

# Structured result for huggingface model loads
struct HuggingFaceLoadResult
    path::Union{String,Nothing}
    info::Any
    downloaded::Bool
end

"""
Download and load a HuggingFace model by name using HuggingFaceHub.jl when
available. Returns a `HuggingFaceLoadResult` with `path`, `info`, and
`downloaded` indicating success.
"""
function load_huggingface_model(model::String; token::Union{Nothing,String}=nothing)
    # Prefer using HuggingFaceHub.jl if it's available
    if isdefined(Main, :HuggingFaceHub)
        try
            HF = Main.HuggingFaceHub
            # Try to use a generic download helper if available
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
                        # Attempt to download the repository snapshot into current dir
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

    # HF not available — return model string in info and mark not downloaded
    @warn "HuggingFaceHub.jl not available — returning model string. Install HuggingFaceHub.jl for direct downloads."
    return HuggingFaceLoadResult(nothing, model, false)
end
function collect_files_with_extensions(directory::String, extensions::Vector{String})
    files = String[]
    for (root, _, file_names) in walkdir(directory)
        for file_name in file_names
            if any(endswith(file_name, ext) for ext in extensions)
                full_path = joinpath(root, file_name)
                push!(files, full_path)
            end
        end
    end
    return files
end

function write_combined_file(files::Vector{String}, output_file::String)
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
    # Infer schema from explicit schema_name or from model strings
    schema_for_generator = get_schema(schema_name, model_name)
    schema_for_embedder = get_schema(schema_name, model_embedding)

    PromptingTools.register_model!(name=model_name, schema=schema_for_generator)
    PromptingTools.register_model!(name=model_embedding, schema=schema_for_embedder)

    PromptingTools.MODEL_CHAT = model_name
    PromptingTools.MODEL_EMBEDDING = model_embedding
end

end
