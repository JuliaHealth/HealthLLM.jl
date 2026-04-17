module Providers

using PromptingTools
using HTTP
using JSON3
using Base: sleep

export ModelProvider, EmbeddingProvider, HuggingFaceProvider, GroqProvider, OllamaProvider, GeminiProvider, OpenAIProvider, AnthropicProvider, HuggingFaceEmbedder

abstract type ModelProvider end
abstract type EmbeddingProvider end

# Abstract methods
function generate(provider::ModelProvider, prompt::String; kwargs...)
    error("generate not implemented for $(typeof(provider))")
end

function embed(provider::EmbeddingProvider, texts::Vector{String}; kwargs...)
    error("embed not implemented for $(typeof(provider))")
end

# Utility function for retries
function with_retry(f::Function, max_retries::Int=3, backoff::Float64=1.0)
    for attempt in 1:max_retries
        try
            return f()
        catch e
            if attempt == max_retries
                rethrow(e)
            end
            if isa(e, HTTP.ExceptionRequest.StatusError) && e.status in [429, 500, 502, 503, 504]
                sleep(backoff * attempt)
            else
                rethrow(e)
            end
        end
    end
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
    with_retry() do
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
end

struct GroqProvider <: ModelProvider
    api_key::String
    model::String
end

function GroqProvider(; api_key=get(ENV, "GROQ_API_KEY", ""), model="llama-3.3-70b-versatile")
    GroqProvider(api_key, model)
end

function generate(provider::GroqProvider, prompt::String; kwargs...)
    with_retry() do
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
end

struct OllamaProvider <: ModelProvider
    model::String
    base_url::String
end

function OllamaProvider(; model="qwen2.5:1.5b", base_url="http://localhost:11434")
    OllamaProvider(model, base_url)
end

function generate(provider::OllamaProvider, prompt::String; kwargs...)
    with_retry() do
        # Register if not
        if !haskey(PromptingTools.MODEL_REGISTRY, provider.model)
            PromptingTools.register_model!(; name=provider.model, schema=PromptingTools.OllamaSchema())
        end
        response = aigenerate(prompt; model=provider.model, api_kwargs=(url=provider.base_url,), kwargs...)
        return response.content
    end
end

struct GeminiProvider <: ModelProvider
    api_key::String
    model::String
end

function GeminiProvider(; api_key=get(ENV, "GEMINI_API_KEY", ""), model="gemini-1.5-flash")
    GeminiProvider(api_key, model)
end

function generate(provider::GeminiProvider, prompt::String; kwargs...)
    with_retry() do
        url = "https://generativelanguage.googleapis.com/v1beta/models/$(provider.model):generateContent?key=$(provider.api_key)"
        headers = ["Content-Type" => "application/json"]
        body = JSON3.write(Dict("contents" => [Dict("parts" => [Dict("text" => prompt)])]))
        response = HTTP.post(url, headers, body)
        data = JSON3.read(response.body)
        return data["candidates"][1]["content"]["parts"][1]["text"]
    end
end

struct OpenAIProvider <: ModelProvider
    api_key::String
    model::String
end

function OpenAIProvider(; api_key=get(ENV, "OPENAI_API_KEY", ""), model="gpt-4")
    OpenAIProvider(api_key, model)
end

function generate(provider::OpenAIProvider, prompt::String; kwargs...)
    with_retry() do
        response = aigenerate(prompt; model=provider.model, api_key=provider.api_key, kwargs...)
        return response.content
    end
end

struct AnthropicProvider <: ModelProvider
    api_key::String
    model::String
end

function AnthropicProvider(; api_key=get(ENV, "ANTHROPIC_API_KEY", ""), model="claude-3-sonnet-20240229")
    AnthropicProvider(api_key, model)
end

function generate(provider::AnthropicProvider, prompt::String; kwargs...)
    with_retry() do
        url = "https://api.anthropic.com/v1/messages"
        headers = [
            "x-api-key" => provider.api_key,
            "anthropic-version" => "2023-06-01",
            "Content-Type" => "application/json"
        ]
        body = JSON3.write(Dict(
            "model" => provider.model,
            "max_tokens" => 1024,
            "messages" => [Dict("role" => "user", "content" => prompt)]
        ))
        response = HTTP.post(url, headers, body)
        data = JSON3.read(response.body)
        return data["content"][1]["text"]
    end
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
    with_retry() do
        url = "https://api-inference.huggingface.co/models/$(provider.model)"
        headers = ["Authorization" => "Bearer $(provider.api_key)", "Content-Type" => "application/json"]
        body = JSON3.write(Dict("inputs" => texts))
        response = HTTP.post(url, headers, body)
        data = JSON3.read(response.body)
        # Assume data is array of embeddings
        return [Vector{Float32}(emb) for emb in data]
    end
end

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