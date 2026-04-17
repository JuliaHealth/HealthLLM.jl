module Providers

using PromptingTools
using HTTP
using JSON3

export ModelProvider, EmbeddingProvider, HuggingFaceProvider, GroqProvider, OllamaProvider

abstract type ModelProvider end
abstract type EmbeddingProvider end

# Abstract methods
function generate(provider::ModelProvider, prompt::String; kwargs...)
    error("generate not implemented for $(typeof(provider))")
end

function embed(provider::EmbeddingProvider, texts::Vector{String}; kwargs...)
    error("embed not implemented for $(typeof(provider))")
end

# Concrete implementations

struct HuggingFaceProvider <: ModelProvider
    api_key::String
    endpoint::String
    model::String
end

function HuggingFaceProvider(; api_key=get(ENV, "HF_TOKEN", ""), endpoint="https://router.huggingface.co/chat/completions", model="Qwen/Qwen3.5-0.8B")
    HuggingFaceProvider(api_key, endpoint, model)
end

function generate(provider::HuggingFaceProvider, prompt::String; kwargs...)
    response = aigenerate(
        PromptingTools.CustomOpenAISchema(),
        prompt;
        model=provider.model,
        api_key=provider.api_key,
        api_kwargs=(; url=provider.endpoint),
        kwargs...
    )
    return response.content
end

struct GroqProvider <: ModelProvider
    api_key::String
    model::String
end

function GroqProvider(; api_key=get(ENV, "GROQ_API_KEY", ""), model="llama-3.3-70b-versatile")
    GroqProvider(api_key, model)
end

function generate(provider::GroqProvider, prompt::String; kwargs...)
    # Register model if not already
    if !haskey(PromptingTools.MODEL_REGISTRY, provider.model)
        PromptingTools.register_model!(; name=provider.model, schema=PromptingTools.CustomOpenAISchema())
    end
    response = aigenerate(
        prompt;
        model=provider.model,
        api_key=provider.api_key,
        api_kwargs=(url="https://api.groq.com/openai/v1/",),
        kwargs...
    )
    return response.content
end

struct OllamaProvider <: ModelProvider
    model::String
    base_url::String
end

function OllamaProvider(; model="qwen2.5:1.5b", base_url="http://localhost:11434")
    OllamaProvider(model, base_url)
end

function generate(provider::OllamaProvider, prompt::String; kwargs...)
    # Register if not
    if !haskey(PromptingTools.MODEL_REGISTRY, provider.model)
        PromptingTools.register_model!(; name=provider.model, schema=PromptingTools.OllamaSchema())
    end
    response = aigenerate(prompt; model=provider.model, api_kwargs=(url=provider.base_url,), kwargs...)
    return response.content
end

# Placeholder for other providers like Gemini, OpenAI, etc.
struct GeminiProvider <: ModelProvider
    api_key::String
    model::String
end

function GeminiProvider(; api_key=get(ENV, "GEMINI_API_KEY", ""), model="gemini-1.5-flash")
    GeminiProvider(api_key, model)
end

function generate(provider::GeminiProvider, prompt::String; kwargs...)
    # Implement Gemini API call
    url = "https://generativelanguage.googleapis.com/v1beta/models/$(provider.model):generateContent?key=$(provider.api_key)"
    headers = ["Content-Type" => "application/json"]
    body = JSON3.write(Dict("contents" => [Dict("parts" => [Dict("text" => prompt)])]))
    response = HTTP.post(url, headers, body)
    data = JSON3.read(response.body)
    return data["candidates"][1]["content"]["parts"][1]["text"]
end

struct OpenAIProvider <: ModelProvider
    api_key::String
    model::String
end

function OpenAIProvider(; api_key=get(ENV, "OPENAI_API_KEY", ""), model="gpt-4")
    OpenAIProvider(api_key, model)
end

function generate(provider::OpenAIProvider, prompt::String; kwargs...)
    response = aigenerate(prompt; model=provider.model, api_key=provider.api_key, kwargs...)
    return response.content
end

# Embedding providers
struct HuggingFaceEmbedder <: EmbeddingProvider
    api_key::String
    model::String
end

function HuggingFaceEmbedder(; api_key=get(ENV, "HF_TOKEN", ""), model="sentence-transformers/all-MiniLM-L6-v2")
    HuggingFaceEmbedder(api_key, model)
end

function embed(provider::HuggingFaceEmbedder, texts::Vector{String}; kwargs...)
    # Implement embedding call
    url = "https://api-inference.huggingface.co/models/$(provider.model)"
    headers = ["Authorization" => "Bearer $(provider.api_key)", "Content-Type" => "application/json"]
    body = JSON3.write(Dict("inputs" => texts))
    response = HTTP.post(url, headers, body)
    data = JSON3.read(response.body)
    # Assume data is array of embeddings
    return [Vector{Float32}(emb) for emb in data]
end

end