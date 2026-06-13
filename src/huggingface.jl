module HuggingFaceLLM

using PromptingTools
using PromptingTools: AbstractPromptSchema, DataMessage
using HuggingFaceHub: infer
import HuggingFaceHub: client, Token as HFToken
import PromptingTools: aiembed, aigenerate

export HuggingFaceManagedSchema, configure_hf_token!, strip_hf_prefix

struct HuggingFaceManagedSchema <: AbstractPromptSchema end

const DEFAULT_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"
const DEFAULT_GENERATION = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

function configure_hf_token!(; token::Union{String,Nothing}=nothing)
    tok = token !== nothing ? token :
          get(ENV, "HF_API_TOKEN", nothing) !== nothing ? ENV["HF_API_TOKEN"] :
          get(ENV, "HUGGING_FACE_HUB_TOKEN", nothing) !== nothing ? ENV["HUGGING_FACE_HUB_TOKEN"] :
          nothing
    if tok !== nothing
        client().token = HFToken(tok)
        return true
    end
    return false
end

strip_hf_prefix(model::AbstractString) = replace(model, r"^hf:" => "")

function aiembed(
    prompt_schema::HuggingFaceManagedSchema,
    doc::AbstractString,
    postprocess::F=identity;
    verbose::Bool=true,
    api_key::String="",
    model::String=DEFAULT_EMBEDDING,
    kwargs...
) where {F<:Function}
    model_id = strip_hf_prefix(model)
    !isempty(api_key) && (client().token = HFToken(api_key))
    configure_hf_token!()
    time = @elapsed begin
        result = infer(model_id, doc; pipeline="feature-extraction")
    end
    emb_matrix = reduce(hcat, result)
    embedding = vec(mean(emb_matrix, dims=2))
    msg = DataMessage(;
        content=postprocess(embedding),
        status=200,
        cost=nothing,
        tokens=(0, 0),
        elapsed=time)
    verbose && @info "HF Embedding ($model_id): $(length(embedding)) dims in $(round(time, digits=2))s"
    return msg
end

function aiembed(
    prompt_schema::HuggingFaceManagedSchema,
    docs::AbstractVector{<:AbstractString},
    postprocess::F=identity;
    verbose::Bool=true,
    api_key::String="",
    model::String=DEFAULT_EMBEDDING,
    kwargs...
) where {F<:Function}
    model_id = strip_hf_prefix(model)
    messages = [
        aiembed(prompt_schema, doc, postprocess;
            verbose=false, api_key, model=model_id, kwargs...)
        for doc in docs
    ]
    msg = DataMessage(;
        content=mapreduce(m -> m.content, hcat, messages),
        status=maximum(m.status for m in messages),
        cost=nothing,
        tokens=(0, 0),
        elapsed=sum(m.elapsed for m in messages))
    verbose && @info "HF Embedding ($model_id): $(length(docs)) docs in $(round(msg.elapsed, digits=2))s"
    return msg
end

function aigenerate(
    prompt_schema::HuggingFaceManagedSchema,
    prompt::AbstractString;
    verbose::Bool=true,
    api_key::String="",
    model::String=DEFAULT_GENERATION,
    max_new_tokens::Int=512,
    temperature::Float64=0.7,
    kwargs...
)
    model_id = strip_hf_prefix(model)
    !isempty(api_key) && (client().token = HFToken(api_key))
    configure_hf_token!()
    time = @elapsed begin
        result = infer(model_id, prompt;
            pipeline="text-generation",
            max_new_tokens=max_new_tokens,
            temperature=temperature)
    end
    generated = if result isa AbstractVector && length(result) > 0
        first(result).generated_text
    else
        string(result)
    end
    text = if length(generated) > length(prompt) && generated[1:length(prompt)] == prompt
        generated[nextind(generated, length(prompt)):end]
    else
        generated
    end
    text = strip(text)
    msg = DataMessage(;
        content=text,
        status=200,
        cost=nothing,
        tokens=(0, 0),
        elapsed=time)
    verbose && @info "HF Generate ($model_id): $(length(text)) chars in $(round(time, digits=2))s"
    return msg
end

function aigenerate(
    prompt_schema::HuggingFaceManagedSchema,
    prompt::AbstractVector;
    verbose::Bool=true,
    api_key::String="",
    model::String=DEFAULT_GENERATION,
    max_new_tokens::Int=512,
    temperature::Float64=0.7,
    kwargs...
)
    rendered = PromptingTools.render(prompt_schema, prompt)
    return aigenerate(prompt_schema, rendered;
        verbose, api_key, model, max_new_tokens, temperature)
end

end
