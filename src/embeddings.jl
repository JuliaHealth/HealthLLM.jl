"""
    Embeddings

Embedding-model selection, generation, and validation for the RAG pipeline.

The default model, [`DEFAULT_EMBEDDING_MODEL`](@ref), is chosen to be usable
from **both** a local [Ollama](https://ollama.com) server and
[HuggingFace](https://huggingface.co), so the same index can be reproduced in
either environment. See [`EMBEDDING_MODELS`](@ref) for the registry.

## Layout

| Piece                        | Responsibility                                  |
|------------------------------|-------------------------------------------------|
| [`EmbeddingModel`](@ref)     | one model's Ollama tag, HF repo, and dimension  |
| [`EMBEDDING_MODELS`](@ref)   | registry of models available on both backends   |
| [`embed`](@ref)              | text → embedding matrix via the chosen backend  |
| [`validate_embeddings`](@ref)| dimension / finiteness / non-degeneracy checks  |
| [`cosine_similarity`](@ref)  | similarity helper for sanity checks             |
| [`embedding_sanity_check`](@ref) | live semantic-ordering check                |
"""
module Embeddings

using PromptingTools
using LinearAlgebra
using ..Utils: get_schema, check_dims

export EmbeddingModel, EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL,
    embedding_model, embedding_ref, embedding_dimension,
    embed, cosine_similarity, similarity_matrix,
    validate_embeddings, embedding_sanity_check

"""
    EmbeddingModel

Describes one embedding model and how to reach it from each backend.

# Fields
- `name::String`: Canonical name (registry key).
- `ollama::String`: Ollama model tag (e.g. `"nomic-embed-text"`).
- `huggingface::String`: HuggingFace repo id (e.g. `"nomic-ai/nomic-embed-text-v1.5"`).
- `dim::Int`: Output embedding dimension.
- `normalized::Bool`: Whether the model emits unit-norm vectors by default.
"""
struct EmbeddingModel
    name::String
    ollama::String
    huggingface::String
    dim::Int
    normalized::Bool
end

"""
    EMBEDDING_MODELS

Registry of embedding models reachable from both Ollama and HuggingFace, keyed by
canonical name. Every entry can be pulled with `ollama pull <tag>` and also exists
as a HuggingFace repo, so an index is reproducible across the two backends.

| Name                | Dim  | Ollama tag          | HuggingFace repo                        |
|---------------------|------|---------------------|-----------------------------------------|
| `nomic-embed-text`  | 768  | `nomic-embed-text`  | `nomic-ai/nomic-embed-text-v1.5`        |
| `all-minilm`        | 384  | `all-minilm`        | `sentence-transformers/all-MiniLM-L6-v2`|
| `mxbai-embed-large` | 1024 | `mxbai-embed-large` | `mixedbread-ai/mxbai-embed-large-v1`    |
| `bge-m3`            | 1024 | `bge-m3`            | `BAAI/bge-m3`                           |
"""
const EMBEDDING_MODELS = Dict{String,EmbeddingModel}(
    "nomic-embed-text" => EmbeddingModel(
        "nomic-embed-text", "nomic-embed-text", "nomic-ai/nomic-embed-text-v1.5", 768, false),
    "all-minilm" => EmbeddingModel(
        "all-minilm", "all-minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, true),
    "mxbai-embed-large" => EmbeddingModel(
        "mxbai-embed-large", "mxbai-embed-large", "mixedbread-ai/mxbai-embed-large-v1", 1024, false),
    "bge-m3" => EmbeddingModel(
        "bge-m3", "bge-m3", "BAAI/bge-m3", 1024, true),
)

"""
    DEFAULT_EMBEDDING_MODEL

Canonical name of the recommended embedding model: `"nomic-embed-text"`. It is
available on both backends, produces 768-dimensional vectors with strong
retrieval quality, and is already the model used throughout the ingestion docs.
For a lighter/faster option use `"all-minilm"` (384-d).
"""
const DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

"""
    embedding_model(name=DEFAULT_EMBEDDING_MODEL) -> EmbeddingModel

Look up an [`EmbeddingModel`](@ref) by canonical name, erroring with the list of
known names if it is not registered.
"""
function embedding_model(name::AbstractString=DEFAULT_EMBEDDING_MODEL)
    m = get(EMBEDDING_MODELS, String(name), nothing)
    m === nothing && throw(ArgumentError(
        "Unknown embedding model '$name'. Known models: " *
        join(sort(collect(keys(EMBEDDING_MODELS))), ", ")))
    return m
end

"""
    embedding_ref(name=DEFAULT_EMBEDDING_MODEL; provider=:ollama) -> String

Return the backend-specific model reference for `name`: the Ollama tag for
`provider=:ollama`, or the `"hf:<repo>"`-prefixed HuggingFace id for
`provider=:huggingface`.
"""
function embedding_ref(name::AbstractString=DEFAULT_EMBEDDING_MODEL; provider::Symbol=:ollama)
    m = embedding_model(name)
    provider === :ollama && return m.ollama
    provider === :huggingface && return "hf:" * m.huggingface
    throw(ArgumentError("provider must be :ollama or :huggingface, got :$provider"))
end

"""
    embedding_dimension(name=DEFAULT_EMBEDDING_MODEL) -> Int

Return the output dimension of the named embedding model.
"""
embedding_dimension(name::AbstractString=DEFAULT_EMBEDDING_MODEL) = embedding_model(name).dim

# Normalise an aiembed result into a dim × n Float32 matrix (columns = chunks).
function _as_matrix(content)
    m = content isa AbstractVector ? reshape(content, :, 1) : content
    return Matrix{Float32}(m)
end

"""
    embed(texts, name=DEFAULT_EMBEDDING_MODEL; provider=:ollama, kwargs...) -> Matrix{Float32}

Embed one or more texts with the named model and return a `dim × n` matrix whose
columns are the embeddings (the row=dimension / column=chunk convention used by
the storage layer and [`validate_embeddings`](@ref)).

`provider` picks the backend (`:ollama` or `:huggingface`); extra `kwargs` are
forwarded to `PromptingTools.aiembed`. Requires the corresponding backend to be
reachable (a running Ollama server, or HuggingFace access).

# Example

```julia
E = embed(["OMOP person table", "FunSQL From clause"])   # 768 × 2
size(E, 1) == embedding_dimension()                       # true
```
"""
function embed(texts::AbstractVector{<:AbstractString},
    name::AbstractString=DEFAULT_EMBEDDING_MODEL; provider::Symbol=:ollama, kwargs...)
    isempty(texts) && throw(ArgumentError("`texts` is empty; nothing to embed."))
    ref = embedding_ref(name; provider=provider)
    res = PromptingTools.aiembed(get_schema(provider), texts; model=ref, kwargs...)
    return _as_matrix(res.content)
end

embed(text::AbstractString, name::AbstractString=DEFAULT_EMBEDDING_MODEL; kwargs...) =
    embed([text], name; kwargs...)

"""
    cosine_similarity(a, b) -> Float64

Cosine similarity between two vectors. Returns `0.0` if either vector is zero.
"""
function cosine_similarity(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) ||
        throw(DimensionMismatch("vectors differ in length: $(length(a)) vs $(length(b))"))
    na, nb = norm(a), norm(b)
    (iszero(na) || iszero(nb)) && return 0.0
    return dot(a, b) / (na * nb)
end

"""
    similarity_matrix(embeddings) -> Matrix{Float64}

Pairwise cosine-similarity matrix over the columns of a `dim × n` embedding
matrix. The diagonal is ~1 for any non-degenerate embedding set.
"""
function similarity_matrix(embeddings::AbstractMatrix)
    n = size(embeddings, 2)
    S = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        S[i, j] = cosine_similarity(view(embeddings, :, i), view(embeddings, :, j))
    end
    return S
end

"""
    validate_embeddings(embeddings; expected_dim=nothing, chunks=nothing, atol=1e-4) -> NamedTuple

Validate an embedding matrix (`dim × n`, columns = chunks) and return a
`(; dim, n, norms)` summary. Checks, in order:

1. non-empty (at least one column);
2. dimension equals `expected_dim` when given;
3. column count equals `length(chunks)` when `chunks` is given;
4. all entries finite (no `NaN`/`Inf`);
5. no zero-vector columns (degenerate embeddings);
6. self-similarity: each column's cosine similarity with itself is `≈ 1` (`atol`).

Throws `ArgumentError`/`DimensionMismatch` on the first failure.

# Example

```julia
E = rand(Float32, 384, 12)
validate_embeddings(E; expected_dim=384)   # (; dim=384, n=12, norms=[...])
```
"""
function validate_embeddings(embeddings::AbstractMatrix;
    expected_dim::Union{Nothing,Integer}=nothing,
    chunks::Union{Nothing,AbstractVector}=nothing, atol::Real=1e-4)

    size(embeddings, 2) == 0 &&
        throw(ArgumentError("No embeddings to validate (matrix has zero columns)."))
    dim, n = check_dims(embeddings, chunks, expected_dim)

    all(isfinite, embeddings) ||
        throw(ArgumentError("Embeddings contain non-finite values (NaN/Inf)."))

    norms = [norm(view(embeddings, :, i)) for i in 1:n]
    zeros_at = findall(iszero, norms)
    isempty(zeros_at) ||
        throw(ArgumentError("Zero-vector (degenerate) embedding column(s): $zeros_at."))

    for i in 1:n
        s = cosine_similarity(view(embeddings, :, i), view(embeddings, :, i))
        isapprox(s, 1.0; atol=atol) ||
            throw(ArgumentError("Self-similarity of column $i is $s, expected ≈ 1."))
    end
    return (; dim, n, norms)
end

"""
    embedding_sanity_check(name=DEFAULT_EMBEDDING_MODEL;
                           provider=:ollama,
                           anchor="How do I count patients in OMOP?",
                           near="Query the number of persons in the CDM",
                           far="A recipe for chocolate cake",
                           kwargs...) -> NamedTuple

Live sanity check that the model embeds meaning, not noise: it embeds an
`anchor`, a semantically *near* sentence, and an unrelated *far* sentence, then
verifies `cos(anchor, near) > cos(anchor, far)`. Returns
`(; near, far, passed, dim)`. Requires the backend to be reachable.
"""
function embedding_sanity_check(name::AbstractString=DEFAULT_EMBEDDING_MODEL;
    provider::Symbol=:ollama,
    anchor::AbstractString="How do I count patients in OMOP?",
    near::AbstractString="Query the number of persons in the CDM",
    far::AbstractString="A recipe for chocolate cake", kwargs...)

    E = embed([anchor, near, far], name; provider=provider, kwargs...)
    validate_embeddings(E; expected_dim=embedding_dimension(name))
    sim_near = cosine_similarity(view(E, :, 1), view(E, :, 2))
    sim_far = cosine_similarity(view(E, :, 1), view(E, :, 3))
    return (; near=sim_near, far=sim_far, passed=sim_near > sim_far, dim=size(E, 1))
end

end
