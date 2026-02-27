module Utils

using PromptingTools

"""
    collect_files_with_extensions(directory, extensions)

Recursively collect file paths under `directory` whose extension is in `extensions`.
Extensions are matched case-insensitively (e.g. `".jl"`).
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
    register_models(model_name, model_embedding; schema=PromptingTools.OllamaSchema())

Register chat and embedding models in PromptingTools and set them as active defaults.
"""
function register_models(
    model_name::AbstractString,
    model_embedding::AbstractString;
    schema=PromptingTools.OllamaSchema()
)
    PromptingTools.register_model!(name=model_name, schema=schema)
    PromptingTools.register_model!(name=model_embedding, schema=schema)

    PromptingTools.MODEL_CHAT = model_name
    PromptingTools.MODEL_EMBEDDING = model_embedding
    return nothing
end

end