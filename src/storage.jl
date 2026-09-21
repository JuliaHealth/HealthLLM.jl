"""
    Storage

Vector-store backends for the RAG pipeline. Every backend shares one interface so
embeddings produced by [`embed`](@ref) can be persisted and queried the
same way regardless of where they live:

| Backend                | Where vectors live               | Extra requirement          |
|------------------------|----------------------------------|----------------------------|
| [`LocalVectorStore`](@ref) | in-memory, `Serialization` to disk | none (built-in)         |
| [`PgVectorStore`](@ref)    | PostgreSQL + pgvector          | `LibPQ` + a live server    |
| [`FaissVectorStore`](@ref) | a FAISS index                  | `Faiss.jl` loaded in `Main`|

## Interface

Each `AbstractVectorStore` supports:

- [`add!`](@ref)`(store, embeddings, chunks)` — append `dim × n` embeddings with their texts;
- [`search`](@ref)`(store, query, k)` — return the `k` nearest [`Hit`](@ref)s;
- `Base.length(store)` — number of stored vectors;
- [`save`](@ref) / [`load`](@ref) — persist and restore where the backend supports it.
"""
module Storage

using LinearAlgebra
using Serialization
import ..Database: store_embeddings_pgvector, search_embeddings_pgvector
using ..Embeddings: cosine_similarity, embed, DEFAULT_EMBEDDING_MODEL
using ..Utils: require_main_module, check_dims, check_k

export AbstractVectorStore, LocalVectorStore, PgVectorStore, FaissVectorStore,
    Hit, add!, search, retrieve, save, load

"""
    AbstractVectorStore

Supertype for all vector-store backends. Concrete stores implement [`add!`](@ref),
[`search`](@ref), and `Base.length`.
"""
abstract type AbstractVectorStore end

"""
    Hit(index, chunk, score, distance=nothing)

One retrieval result, in the shape every backend returns. A uniform hit means
downstream code — prompt building above all — never has to know which store the
chunk came from.

# Fields
- `index::Int`: Position of the vector in the store, or the database row id for
  [`PgVectorStore`](@ref).
- `chunk::String`: The stored text.
- `score::Float64`: Similarity, **larger is always more similar**, regardless of
  the backend's native metric. Cosine backends report cosine similarity in
  `[-1, 1]`; distance-based ones report a negated distance.
- `distance::Union{Float64,Nothing}`: The backend's raw distance where it has
  one (pgvector), else `nothing`.
"""
struct Hit
    index::Int
    chunk::String
    score::Float64
    distance::Union{Float64,Nothing}
end

Hit(index::Integer, chunk::AbstractString, score::Real) =
    Hit(Int(index), String(chunk), Float64(score), nothing)

Base.show(io::IO, h::Hit) =
    print(io, "Hit(", h.index, ", score=", round(h.score, digits=3), ", ",
        repr(first(h.chunk, 40)), ")")

"""
    retrieve(store, query::AbstractString, k=5;
             model=DEFAULT_EMBEDDING_MODEL, provider=:ollama, embedder=embed, kwargs...)

Per-query retrieval: embed a natural-language `query` with `model` and return the
`k` nearest stored chunks from `store`. This is the text-in entry point that pairs
[`embed`](@ref) with a store's [`search`](@ref) — whereas [`search`](@ref) takes a
pre-computed query *vector*, `retrieve` takes the *string* and handles embedding.

Works with any [`AbstractVectorStore`](@ref); the returned hits are exactly what the
backend's [`search`](@ref) yields — a `Vector{`[`Hit`](@ref)`}` whatever the backend.
`provider` and extra `kwargs` are forwarded
to the embedder. Pass `embedder` to substitute the embedding function (useful for
tests or a cached embedder); it is called as `embedder(query, model; provider, kwargs...)`
and must return the query embedding as a `dim × 1` matrix or a length-`dim` vector.

# Example

```julia
store = LocalVectorStore(embedding_dimension())
add!(store, embed(chunks), chunks)

hits = retrieve(store, "How do I count patients in OMOP?", 5)
for h in hits
    println(round(h.score, digits=3), "  ", first(h.chunk, 80))
end
```
"""
function retrieve(store::AbstractVectorStore, query::AbstractString, k::Integer=5;
    model::AbstractString=DEFAULT_EMBEDDING_MODEL, provider::Symbol=:ollama,
    embedder=embed, kwargs...)
    isempty(query) && throw(ArgumentError("`query` is empty; nothing to retrieve."))
    E = embedder(query, model; provider=provider, kwargs...)
    qvec = E isa AbstractMatrix ? vec(view(E, :, 1)) : vec(collect(E))
    return search(store, qvec, k)
end

# ---------------------------------------------------------------------------
# LocalVectorStore
# ---------------------------------------------------------------------------

"""
    LocalVectorStore(dim; normalize=true)

In-memory, file-backed vector store. Embeddings are held as a `dim × n` matrix and
searched with cosine similarity — no external service required. Use [`save`](@ref)
and [`load`](@ref) to round-trip the store through `Serialization`.

# Fields
- `dim::Int`: Embedding dimension every added vector must match.
- `embeddings::Matrix{Float32}`: `dim × n` matrix, columns = stored vectors.
- `chunks::Vector{String}`: Text for each column, aligned with `embeddings`.
- `normalize::Bool`: Store L2-normalised vectors (makes cosine search a dot product).

# Example

```julia
store = LocalVectorStore(768)
add!(store, embed(["OMOP person table", "FunSQL From clause"]),
             ["OMOP person table", "FunSQL From clause"])
hits = search(store, embed("count patients"), 1)
hits[1].chunk        # "OMOP person table"
save(store, "index.jls")
store2 = load(LocalVectorStore, "index.jls")
```
"""
mutable struct LocalVectorStore <: AbstractVectorStore
    dim::Int
    embeddings::Matrix{Float32}
    chunks::Vector{String}
    normalize::Bool
end

function LocalVectorStore(dim::Integer; normalize::Bool=true)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    return LocalVectorStore(Int(dim), Matrix{Float32}(undef, dim, 0), String[], normalize)
end

Base.length(store::LocalVectorStore) = length(store.chunks)

_normalize_cols(M::AbstractMatrix) = mapslices(c -> (n = norm(c); iszero(n) ? c : c ./ n), M; dims=1)

"""
    add!(store::LocalVectorStore, embeddings, chunks) -> store

Append a `dim × n` embedding matrix and its `n` text chunks to `store`. Vectors are
L2-normalised first when `store.normalize` is set. Errors if the dimension or the
column/chunk counts do not line up.
"""
function add!(store::LocalVectorStore, embeddings::AbstractMatrix, chunks::AbstractVector)
    check_dims(embeddings, chunks, store.dim)
    E = Matrix{Float32}(embeddings)
    store.normalize && (E = Matrix{Float32}(_normalize_cols(E)))
    store.embeddings = hcat(store.embeddings, E)
    append!(store.chunks, String.(chunks))
    return store
end

"""
    search(store::LocalVectorStore, query, k=5) -> Vector{Hit}

Return the `k` most cosine-similar chunks to `query` (a length-`dim` vector), as
[`Hit`](@ref)s ordered most-similar first. `score` is the cosine similarity in
`[-1, 1]`.
"""
function search(store::LocalVectorStore, query::AbstractVector, k::Integer=5)
    check_k(k)
    length(query) == store.dim ||
        throw(DimensionMismatch("query length ($(length(query))) != store dim ($(store.dim))"))
    n = length(store)
    n == 0 && return Hit[]
    scores = [cosine_similarity(view(store.embeddings, :, i), query) for i in 1:n]
    order = partialsortperm(scores, 1:min(k, n); rev=true)
    return [Hit(i, store.chunks[i], scores[i]) for i in order]
end

"""
    save(store::LocalVectorStore, path) -> path

Serialise `store` to `path` (via `Serialization`). Restore with
[`load`](@ref)`(LocalVectorStore, path)`.
"""
function save(store::LocalVectorStore, path::AbstractString)
    open(path, "w") do io
        serialize(io, store)
    end
    return path
end

"""
    load(::Type{LocalVectorStore}, path) -> LocalVectorStore

Load a store previously written with [`save`](@ref).
"""
function load(::Type{LocalVectorStore}, path::AbstractString)
    store = open(deserialize, path)
    store isa LocalVectorStore ||
        throw(ArgumentError("$path does not contain a LocalVectorStore (got $(typeof(store)))"))
    return store
end

# ---------------------------------------------------------------------------
# PgVectorStore
# ---------------------------------------------------------------------------

"""
    PgVectorStore(conn, dim; table="embeddings", metric=:cosine)

PostgreSQL/pgvector-backed store. Delegates persistence to
[`store_embeddings_pgvector`](@ref) and retrieval to
[`search_embeddings_pgvector`](@ref); vectors live in `table` in the
database `conn` points at.

# Fields
- `conn::LibPQ.Connection`: Open connection to a pgvector-enabled database.
- `dim::Int`: Embedding dimension for the table's `VECTOR(dim)` column.
- `table::String`: Destination table name (validated as a SQL identifier).
- `metric::Symbol`: Distance metric for search — `:cosine`, `:dot`, or `:l2`.

# Example

```julia
using LibPQ
conn  = LibPQ.Connection("postgresql://localhost/health")
store = PgVectorStore(conn, 768)
add!(store, embed(chunks), chunks)
hits  = search(store, embed("count patients"), 5)   # (; id, chunk, distance)
```
"""
mutable struct PgVectorStore <: AbstractVectorStore
    conn::Any
    dim::Int
    table::String
    metric::Symbol
end

function PgVectorStore(conn, dim::Integer; table::AbstractString="embeddings", metric::Symbol=:cosine)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    return PgVectorStore(conn, Int(dim), String(table), metric)
end

"""
    add!(store::PgVectorStore, embeddings, chunks) -> store

Create `store.table` if needed and insert the `dim × n` embeddings with their
chunks, in a single transaction. See [`store_embeddings_pgvector`](@ref).
"""
function add!(store::PgVectorStore, embeddings::AbstractMatrix, chunks::AbstractVector)
    store_embeddings_pgvector(store.conn, embeddings, chunks, store.dim; table=store.table)
    return store
end

"""
    search(store::PgVectorStore, query, k=5) -> Vector{Hit}

Return the `k` nearest stored chunks to `query` as [`Hit`](@ref)s, ordered
most-similar first using `store.metric`. The row id lands in `index` and the raw
pgvector distance in `distance`; `score` is that distance flipped to the
larger-is-more-similar convention every backend shares (`1 - d` for cosine, `-d`
for dot and L2). See [`search_embeddings_pgvector`](@ref).
"""
function search(store::PgVectorStore, query::AbstractVector, k::Integer=5)
    rows = search_embeddings_pgvector(store.conn, query, k;
        table=store.table, metric=store.metric)
    return [Hit(row.id, row.chunk, _pg_score(store.metric, row.distance), Float64(row.distance))
            for row in rows]
end

# pgvector distances all order ascending; map them onto the shared
# larger-is-more-similar `score`. Cosine distance is 1 - cos, so 1 - d recovers
# the similarity; dot (`<#>`) and L2 (`<->`) are simply negated.
_pg_score(metric::Symbol, d) = metric === :cosine ? 1.0 - Float64(d) : -Float64(d)

# ---------------------------------------------------------------------------
# FaissVectorStore
# ---------------------------------------------------------------------------

"""
    FaissVectorStore(dim; metric=:cosine)

FAISS-backed store for large local indexes. FAISS is an **optional** dependency:
the store only works when a `Faiss` module is loaded into `Main` (e.g.
`using Faiss` in the driver session) — otherwise construction raises a clear error.
Chunks are kept alongside the index so search can return their text.

# Fields
- `dim::Int`: Embedding dimension.
- `metric::Symbol`: `:cosine`/`:dot` (inner-product index) or `:l2` (L2 index).
- `index::Any`: The underlying FAISS index object.
- `chunks::Vector{String}`: Text aligned with the vectors added to `index`.
- `normalize::Bool`: L2-normalise vectors (so inner product == cosine).

# Example

```julia
using Faiss                       # optional, must be loaded first
store = FaissVectorStore(768)
add!(store, embed(chunks), chunks)
hits = search(store, embed("count patients"), 5)   # (; index, chunk, score)
```
"""
mutable struct FaissVectorStore <: AbstractVectorStore
    dim::Int
    metric::Symbol
    index::Any
    chunks::Vector{String}
    normalize::Bool
end

_faiss_module() = require_main_module(:Faiss,
    "The FAISS backend requires Faiss.jl: install it and run `using Faiss` " *
    "before constructing a FaissVectorStore.")

function FaissVectorStore(dim::Integer; metric::Symbol=:cosine)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    metric in (:cosine, :dot, :l2) ||
        throw(ArgumentError("metric must be :cosine, :dot, or :l2, got :$metric"))
    Faiss = _faiss_module()
    faiss_metric = metric === :l2 ? "L2" : "IP"
    index = if isdefined(Faiss, :Index)
        Faiss.Index(dim; str=faiss_metric)
    else
        throw(ArgumentError("Loaded Faiss module has no `Index` constructor."))
    end
    return FaissVectorStore(Int(dim), metric, index, String[], metric !== :l2)
end

Base.length(store::FaissVectorStore) = length(store.chunks)

"""
    add!(store::FaissVectorStore, embeddings, chunks) -> store

Add `dim × n` embeddings (normalised when `store.normalize`) and their chunks to the
FAISS index. FAISS expects one row per vector, so the `dim × n` matrix is transposed
to `n × dim` before being handed to `Faiss.add`.
"""
function add!(store::FaissVectorStore, embeddings::AbstractMatrix, chunks::AbstractVector)
    check_dims(embeddings, chunks, store.dim)
    Faiss = _faiss_module()
    E = Matrix{Float32}(embeddings)
    store.normalize && (E = Matrix{Float32}(_normalize_cols(E)))
    Faiss.add(store.index, permutedims(E))
    append!(store.chunks, String.(chunks))
    return store
end

"""
    search(store::FaissVectorStore, query, k=5) -> Vector{Hit}

Return the `k` nearest chunks to `query` as [`Hit`](@ref)s. `score` is the raw FAISS
score (inner product for `:cosine`/`:dot`, negative L2 distance for `:l2`, so larger
is always more similar).
"""
function search(store::FaissVectorStore, query::AbstractVector, k::Integer=5)
    check_k(k)
    length(query) == store.dim ||
        throw(DimensionMismatch("query length ($(length(query))) != store dim ($(store.dim))"))
    n = length(store)
    n == 0 && return Hit[]
    Faiss = _faiss_module()
    q = Matrix{Float32}(reshape(collect(query), :, 1))
    store.normalize && (q = Matrix{Float32}(_normalize_cols(q)))
    D, I = Faiss.search(store.index, permutedims(q), min(k, n))
    ids = vec(I)
    dists = vec(D)
    hits = Hit[]
    for (rank, id) in enumerate(ids)
        idx = Int(id) + 1
        (idx < 1 || idx > n) && continue
        score = store.metric === :l2 ? -dists[rank] : dists[rank]
        push!(hits, Hit(idx, store.chunks[idx], score))
    end
    return hits
end

end
