"""
    HuggingFace

HuggingFace backend for PromptingTools.

PromptingTools ships schemas for a long list of OpenAI-compatible providers
(Groq, Together, Fireworks, DeepSeek, Mistral, ...) but — as of v0.94, the latest
release — **none for HuggingFace**. This module supplies the missing one, built
the same way PromptingTools builds its own provider schemas: a marker type under
`AbstractOpenAISchema` plus `create_chat`/`create_embeddings` methods that point
the shared OpenAI transport at HuggingFace's router.

Because [`HuggingFaceOpenAISchema`](@ref) is an `AbstractOpenAISchema`, everything
PromptingTools already does for OpenAI — `aigenerate`, `aiembed`, message
rendering, streaming, retries — works against HuggingFace unchanged.

## Credentials

The token is read from [`huggingface_api_key`](@ref): an explicitly passed
`api_key` wins, then a key set with [`set_huggingface_api_key!`](@ref), then the
first of `HF_API_TOKEN`, `HF_TOKEN`, `HUGGINGFACE_API_KEY`, or
`HUGGING_FACE_HUB_TOKEN` found in the environment.

## Example

```julia
using HealthLLM, PromptingTools

set_huggingface_api_key!(ENV["HF_TOKEN"])

msg = PromptingTools.aigenerate(HuggingFaceOpenAISchema(), "Say hi";
                                model = "meta-llama/Llama-3.1-8B-Instruct")

emb = PromptingTools.aiembed(HuggingFaceOpenAISchema(), ["some text"];
                             model = "BAAI/bge-m3")

# or point embeddings at a TEI / Inference Endpoint that speaks OpenAI
emb = PromptingTools.aiembed(HuggingFaceOpenAISchema(), ["some text"];
                             model = "BAAI/bge-m3",
                             api_kwargs = (; url = "https://my-endpoint.hf.space/v1"))
```

## Endpoints

Chat and embeddings use different HuggingFace surfaces, because the router's
OpenAI-compatible API covers chat only:

| Call        | Endpoint                                                     |
|-------------|--------------------------------------------------------------|
| `aigenerate`| [`HUGGINGFACE_ROUTER_URL`](@ref) `/chat/completions`          |
| `aiembed`   | [`HUGGINGFACE_INFERENCE_URL`](@ref) `/<model>/pipeline/feature-extraction` |
"""
module HuggingFace

import PromptingTools
import OpenAI
using HTTP
using JSON3

export HuggingFaceOpenAISchema, huggingface_api_key, set_huggingface_api_key!,
    huggingface_providers, HUGGINGFACE_ROUTER_URL, HUGGINGFACE_INFERENCE_URL,
    HUGGINGFACE_EMBED_TIMEOUT

"""
    HuggingFaceOpenAISchema

Schema for HuggingFace models served over an OpenAI-compatible API — the
HuggingFace Inference Providers router by default, or any Text Embeddings
Inference / Inference Endpoint deployment you point it at.

A subtype of `PromptingTools.AbstractOpenAISchema`, so it plugs into `aigenerate`
and `aiembed` exactly like the built-in provider schemas. Override the endpoint
per call with `api_kwargs = (; url = "...")`.

# Example

```julia
PromptingTools.aigenerate(HuggingFaceOpenAISchema(), "Hello";
                          model = "meta-llama/Llama-3.1-8B-Instruct")
```
"""
struct HuggingFaceOpenAISchema <: PromptingTools.AbstractOpenAISchema end

"""
    HUGGINGFACE_ROUTER_URL

Base URL of the HuggingFace Inference Providers router (`https://router.huggingface.co/v1`),
the OpenAI-compatible entry point used when no `url` is supplied.
"""
const HUGGINGFACE_ROUTER_URL = "https://router.huggingface.co/v1"

# Env vars checked in order. HF_API_TOKEN is what this repository's heavy tests
# already use; HF_TOKEN is the current HuggingFace CLI default; the other two are
# older spellings still common in CI configs.
const _TOKEN_ENV_VARS = ("HF_API_TOKEN", "HF_TOKEN", "HUGGINGFACE_API_KEY",
    "HUGGING_FACE_HUB_TOKEN")

const _API_KEY = Ref{String}("")

"""
    set_huggingface_api_key!(key) -> String

Set the HuggingFace token for the current session, taking precedence over the
environment. Pass `""` to clear it and fall back to the environment again.
"""
function set_huggingface_api_key!(key::AbstractString)
    _API_KEY[] = String(key)
    return _API_KEY[]
end

"""
    huggingface_api_key() -> String

Return the HuggingFace token: the one set by [`set_huggingface_api_key!`](@ref)
if any, else the first of `HF_API_TOKEN`, `HF_TOKEN`, `HUGGINGFACE_API_KEY`,
`HUGGING_FACE_HUB_TOKEN` present in the environment, else `""`.

An empty result is not an error here — some endpoints are open — but the request
will fail with a 401 if the model requires authentication.
"""
function huggingface_api_key()
    isempty(_API_KEY[]) || return _API_KEY[]
    for var in _TOKEN_ENV_VARS
        key = get(ENV, var, "")
        isempty(key) || return String(key)
    end
    return ""
end

# PromptingTools defaults `api_key` to OPENAI_API_KEY, so a non-empty value here
# does not mean the caller meant it for HuggingFace. Honour anything that is
# genuinely caller-supplied, otherwise reach for the HuggingFace token.
function _resolve_api_key(api_key::AbstractString)
    supplied = !isempty(api_key) && String(api_key) != String(PromptingTools.OPENAI_API_KEY)
    supplied && return String(api_key)
    hf = huggingface_api_key()
    return isempty(hf) ? String(api_key) : hf
end

"""
    hf_model_id(model) -> String

Strip the `"hf:"` prefix this package uses to mark HuggingFace references
(see `embedding_ref`), leaving the bare repo id the API expects.

# Example

```julia
hf_model_id("hf:BAAI/bge-m3")   # "BAAI/bge-m3"
hf_model_id("BAAI/bge-m3")      # "BAAI/bge-m3"
```
"""
hf_model_id(model::AbstractString) =
    startswith(lowercase(String(model)), "hf:") ? String(model)[4:end] : String(model)

"""
    split_provider(model) -> (repo, provider)

Split a `"org/repo:provider"` reference into its parts, returning an empty
`provider` when the model is not pinned. HuggingFace repo ids never contain `:`,
so the last colon unambiguously marks a provider pin.

# Example

```julia
split_provider("Qwen/Qwen2.5-7B-Instruct:featherless-ai")
# ("Qwen/Qwen2.5-7B-Instruct", "featherless-ai")
```
"""
function split_provider(model::AbstractString)
    s = String(model)
    i = findlast(==(':'), s)
    i === nothing && return (s, "")
    return (s[1:prevind(s, i)], s[nextind(s, i):end])
end

"""
    huggingface_providers(model; api_key=huggingface_api_key()) -> Vector{String}

Names of the inference providers currently serving `model`, i.e. those whose
mapping status is `"live"`. Returns an empty vector when nothing serves it.

The router only auto-routes to providers **enabled on your account**, so a model
can be live somewhere and still be refused. Use this to see the options, then pin
one by appending it to the model name.

# Example

```julia
huggingface_providers("Qwen/Qwen2.5-7B-Instruct")   # ["featherless-ai"]
# then: model = "hf:Qwen/Qwen2.5-7B-Instruct:featherless-ai"
```
"""
function huggingface_providers(model::AbstractString;
    api_key::AbstractString=huggingface_api_key())
    repo, _ = split_provider(hf_model_id(model))
    url = "https://huggingface.co/api/models/$repo?expand%5B%5D=inferenceProviderMapping"
    headers = isempty(api_key) ? Pair{String,String}[] :
              ["Authorization" => "Bearer $api_key"]
    resp = HTTP.get(url, headers; status_exception=true, readtimeout=30)
    mapping = get(JSON3.read(String(resp.body)), :inferenceProviderMapping, nothing)
    mapping === nothing && return String[]
    return String[String(name) for (name, info) in pairs(mapping)
                  if String(get(info, :status, "")) == "live"]
end

# HuggingFace answers an unroutable model with a bare "not supported by any
# provider you have enabled", which does not say that the model *is* served, just
# not by a provider this account has switched on. Name the live providers and the
# pinning syntax so the failure is one edit from fixed.
_is_model_not_supported(err) = occursin("model_not_supported", sprint(showerror, err))

function _provider_hint(model::AbstractString, api_key::AbstractString)
    repo, pinned = split_provider(model)
    isempty(pinned) || return nothing        # already pinned; the hint would be noise
    providers = try
        huggingface_providers(repo; api_key=api_key)
    catch
        return nothing
    end
    isempty(providers) && return nothing
    served = length(providers) == 1 ?
             "$(only(providers)), which is not enabled for your account" :
             "$(join(providers, ", ")), none of which is enabled for your account"
    return ErrorException(
        "HuggingFace refused to route '$repo': it is served by $served. " *
        "Pin a provider on the model name, e.g. model = \"hf:$repo:$(first(providers))\", " *
        "or enable one at https://huggingface.co/settings/inference-providers.")
end

function OpenAI.create_chat(schema::HuggingFaceOpenAISchema,
    api_key::AbstractString,
    model::AbstractString,
    conversation;
    url::String=HUGGINGFACE_ROUTER_URL,
    kwargs...)
    key = _resolve_api_key(api_key)
    id = hf_model_id(model)
    try
        return OpenAI.create_chat(PromptingTools.CustomOpenAISchema(),
            key, id, conversation; url, kwargs...)
    catch err
        if _is_model_not_supported(err)
            hint = _provider_hint(id, key)
            hint === nothing || throw(hint)
        end
        rethrow()
    end
end

"""
    HUGGINGFACE_INFERENCE_URL

Base URL for HuggingFace pipeline inference
(`https://router.huggingface.co/hf-inference/models`). Embeddings go here rather
than through [`HUGGINGFACE_ROUTER_URL`](@ref): the router's OpenAI-compatible
surface covers chat completions only and answers `/v1/embeddings` with a 404.
"""
const HUGGINGFACE_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models"

"""
    HUGGINGFACE_EMBED_TIMEOUT

Read timeout in seconds (`300`) used for feature-extraction when the caller did
not choose one. A HuggingFace model that is not already warm loads while holding
the connection open — measured at ~55s for `BAAI/bge-m3` — so PromptingTools'
120s `aiembed` default times out on the first call to a cold large model.
"""
const HUGGINGFACE_EMBED_TIMEOUT = 300

# `aiembed`'s exact default. Matching the whole tuple means we only substitute a
# longer timeout when the caller passed no `http_kwargs` at all; any deliberate
# choice, including a 120s one, is left untouched.
const _AIEMBED_DEFAULT_HTTP_KWARGS = (retry_non_idempotent=true, retries=5, readtimeout=120)

_embed_http_kwargs(http_kwargs::NamedTuple) =
    http_kwargs == _AIEMBED_DEFAULT_HTTP_KWARGS ?
    merge(http_kwargs, (; readtimeout=HUGGINGFACE_EMBED_TIMEOUT)) : http_kwargs

"""
    OpenAI.create_embeddings(schema::HuggingFaceOpenAISchema, api_key, docs, model; url="", kwargs...)

Embed `docs` with a HuggingFace model.

With no `url`, this calls the `feature-extraction` pipeline under
[`HUGGINGFACE_INFERENCE_URL`](@ref) and reshapes the reply into the OpenAI
embeddings response `aiembed` expects, so the HuggingFace backend behaves like
any other from the caller's side. Token-level output (a matrix per input, from
models that do not pool internally) is mean-pooled into one vector per input.

Pass `api_kwargs = (; url = "https://.../v1")` to target a deployment that *does*
serve an OpenAI-compatible `/embeddings` route — a Text Embeddings Inference
container or a dedicated Inference Endpoint — and the request is forwarded there
unchanged instead.
"""
function OpenAI.create_embeddings(schema::HuggingFaceOpenAISchema,
    api_key::AbstractString,
    docs,
    model::AbstractString;
    url::AbstractString="",
    http_kwargs::NamedTuple=NamedTuple(),
    kwargs...)
    key = _resolve_api_key(api_key)
    id = hf_model_id(model)
    isempty(url) || return OpenAI.create_embeddings(PromptingTools.CustomOpenAISchema(),
        key, docs, id; url=String(url), http_kwargs, kwargs...)
    return _feature_extraction(key, docs, id; http_kwargs=http_kwargs)
end

function _feature_extraction(api_key::AbstractString, docs, model::AbstractString;
    http_kwargs::NamedTuple=NamedTuple())
    texts = docs isa AbstractString ? [String(docs)] : String[String(d) for d in docs]
    isempty(texts) && throw(ArgumentError("`docs` is empty; nothing to embed."))

    url = string(HUGGINGFACE_INFERENCE_URL, "/", model, "/pipeline/feature-extraction")
    headers = ["Authorization" => "Bearer $api_key", "Content-Type" => "application/json"]
    # wait_for_model keeps a cold model from failing the first call outright; the
    # cost is that the load time is spent on this open connection, hence the
    # larger default timeout in `_embed_http_kwargs`.
    body = JSON3.write(Dict("inputs" => texts,
        "options" => Dict("wait_for_model" => true)))

    resp = try
        HTTP.post(url, headers, body; status_exception=true, _embed_http_kwargs(http_kwargs)...)
    catch err
        err isa HTTP.Exceptions.TimeoutError && throw(ErrorException(
            "HuggingFace feature-extraction timed out for '$model'. A cold model can " *
            "take minutes to load; retry, or raise the limit with " *
            "`api_kwargs = (; http_kwargs = (; readtimeout = 600))`."))
        rethrow()
    end
    parsed = JSON3.read(String(resp.body))
    vectors = [_pool(item) for item in parsed]

    # Shaped to match OpenAI's embeddings response so `aiembed` needs no special
    # case: it reads `.response[:data][i][:embedding]` and `.status`.
    return (;
        response=Dict(
            :data => [Dict(:embedding => v) for v in vectors],
            :usage => Dict(:prompt_tokens => 0)),
        status=resp.status)
end

# feature-extraction returns one vector per input for models that pool
# internally, and a token × hidden matrix for those that do not; average the
# token axis so callers always get a single vector per input.
function _pool(item)
    isempty(item) && return Float64[]
    first(item) isa Number && return Float64[Float64(x) for x in item]
    rows = [_pool(row) for row in item]
    width = length(first(rows))
    all(r -> length(r) == width, rows) ||
        throw(ArgumentError("ragged embedding rows returned by feature-extraction"))
    return [sum(r[i] for r in rows) / length(rows) for i in 1:width]
end

end
