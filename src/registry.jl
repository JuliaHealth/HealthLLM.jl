module Registry

using ..Providers

const MODEL_PROVIDERS = Dict{String, ModelProvider}()
const EMBEDDING_PROVIDERS = Dict{String, EmbeddingProvider}()

function register_provider(name::String, provider::ModelProvider)
    MODEL_PROVIDERS[name] = provider
end

function register_provider(name::String, provider::EmbeddingProvider)
    EMBEDDING_PROVIDERS[name] = provider
end

function get_provider(name::String)
    if haskey(MODEL_PROVIDERS, name)
        return MODEL_PROVIDERS[name]
    elseif haskey(EMBEDDING_PROVIDERS, name)
        return EMBEDDING_PROVIDERS[name]
    else
        error("Provider $name not found")
    end
end

function list_providers()
    return keys(MODEL_PROVIDERS), keys(EMBEDDING_PROVIDERS)
end

end