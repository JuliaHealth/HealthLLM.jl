```@meta
CurrentModule = HealthLLM
```

# Building Embeddings

Retrieval quality in the RAG pipeline rests on two choices: **which model turns
text into vectors**, and **where those vectors are stored and searched**. This
page covers both — picking a model, generating embeddings, validating them, and
persisting them to one of three interchangeable stores.

```
chunks ─▶ embed ─▶ dim × n matrix ─▶ validate_embeddings ─┐
                                                          ├─▶ LocalVectorStore  (file)
                                                          ├─▶ PgVectorStore     (pgvector)
                                                          └─▶ FaissVectorStore  (FAISS)
```

Throughout, embeddings use the **`dim × n` convention**: one column per chunk,
one row per embedding dimension. This matches the storage layer and
[`validate_embeddings`](@ref).

## Choosing a model

The pipeline is built around embedding models that are reachable from **both** a
local [Ollama](https://ollama.com) server and [HuggingFace](https://huggingface.co),
so an index built in one environment can be reproduced in the other. The registry
is [`EMBEDDING_MODELS`](@ref):

```julia
using HealthLLM

collect(keys(EMBEDDING_MODELS))
# ["nomic-embed-text", "all-minilm", "mxbai-embed-large", "bge-m3"]

embedding_dimension()                    # 768  (the default model)
embedding_dimension("all-minilm")        # 384
```

| Name                | Dim  | Ollama tag          | HuggingFace repo                          | Notes                        |
|---------------------|------|---------------------|-------------------------------------------|------------------------------|
| `nomic-embed-text`  | 768  | `nomic-embed-text`  | `nomic-ai/nomic-embed-text-v1.5`          | **Default** — balanced quality/size |
| `all-minilm`        | 384  | `all-minilm`        | `sentence-transformers/all-MiniLM-L6-v2`  | Lightest / fastest           |
| `mxbai-embed-large` | 1024 | `mxbai-embed-large` | `mixedbread-ai/mxbai-embed-large-v1`      | Higher quality, larger       |
| `bge-m3`            | 1024 | `bge-m3`            | `BAAI/bge-m3`                             | Multilingual, long-context   |

[`DEFAULT_EMBEDDING_MODEL`](@ref) is `"nomic-embed-text"`: it runs on both backends,
produces 768-dimensional vectors with strong retrieval quality, and is the model
used throughout the [ingestion docs](ingestion.md). Reach for `"all-minilm"` when
you want a smaller, faster index and can trade a little accuracy.

Resolve a model's backend-specific reference with [`embedding_ref`](@ref):

```julia
embedding_ref("nomic-embed-text"; provider=:ollama)        # "nomic-embed-text"
embedding_ref("all-minilm";       provider=:huggingface)   # "hf:sentence-transformers/all-MiniLM-L6-v2"
```

!!! note "Backend setup"
    For Ollama, pull the tag once with `ollama pull nomic-embed-text` and make
    sure the server is running. For HuggingFace, set a token (see below). Either
    way, register the model in the RAG pipeline with `register_models(...)` as
    shown in [Getting Started](getting-started.md).

### The HuggingFace backend

PromptingTools ships schemas for a long list of OpenAI-compatible providers but
none for HuggingFace, so this package supplies one:
[`HuggingFaceOpenAISchema`](@ref). It is an `AbstractOpenAISchema`, so
`aigenerate`, `aiembed`, message rendering and retries all work unchanged —
`provider = :huggingface` and `hf:`-prefixed model names route through it
automatically.

Set a token once per session, or let it come from the environment
(`HF_API_TOKEN`, `HF_TOKEN`, `HUGGINGFACE_API_KEY`, `HUGGING_FACE_HUB_TOKEN`,
checked in that order):

```julia
set_huggingface_api_key!(ENV["HF_TOKEN"])
huggingface_api_key()          # what will actually be sent
```

!!! note "Pinning an inference provider"
    The router auto-routes only to providers **enabled on your account**, so a
    model can be live on HuggingFace and still be refused with
    `model_not_supported`. Append the provider to pin it:

    ```julia
    huggingface_providers("Qwen/Qwen2.5-7B-Instruct")   # ["featherless-ai"]

    register_models("hf:Qwen/Qwen2.5-7B-Instruct:featherless-ai", "hf:BAAI/bge-m3")
    ```

    When routing is refused, the error raised here already names the live
    providers and the exact model string to use, so you rarely need to look this
    up yourself. Alternatively enable the provider at
    [huggingface.co/settings/inference-providers](https://huggingface.co/settings/inference-providers).

Chat and embeddings use different HuggingFace surfaces, because the router's
OpenAI-compatible API covers chat completions only — `/v1/embeddings` answers
404 there:

| Call         | Endpoint                                                                  |
|--------------|---------------------------------------------------------------------------|
| `aigenerate` | [`HUGGINGFACE_ROUTER_URL`](@ref) + `/chat/completions`                     |
| `aiembed`    | [`HUGGINGFACE_INFERENCE_URL`](@ref) + `/<model>/pipeline/feature-extraction` |

The feature-extraction reply is reshaped into the OpenAI embeddings response
`aiembed` expects, and token-level output (from models that do not pool
internally) is mean-pooled to one vector per input — so a HuggingFace embedding
matrix is the same `dim × n` shape as an Ollama one.

!!! note "Cold models"
    A HuggingFace model that is not already warm loads while holding the
    connection open — around 40–55s for `bge-m3` in practice. That exceeds
    PromptingTools' 120s `aiembed` default under load, so this package raises the
    read timeout to [`HUGGINGFACE_EMBED_TIMEOUT`](@ref) (300s) when you have not
    chosen one yourself. Any `http_kwargs` you pass is left exactly as given.

To use a deployment that *does* speak OpenAI embeddings — a Text Embeddings
Inference container or a dedicated Inference Endpoint — pass its URL and the
request is forwarded there instead:

```julia
E = embed(chunks, "bge-m3"; provider = :huggingface,
          api_kwargs = (; url = "https://my-endpoint.hf.space/v1"))
```

## Generating embeddings

[`embed`](@ref) turns text into a `dim × n` matrix using the chosen model and
backend. It accepts a single string or a vector of strings:

```julia
E = embed(["OMOP person table", "FunSQL From clause"])   # 768 × 2 (Ollama, default model)
size(E, 1) == embedding_dimension()                       # true

# A lighter model, via HuggingFace:
E2 = embed(chunks, "all-minilm"; provider=:huggingface)
```

`provider` selects the backend (`:ollama` default, or `:huggingface`); any extra
keyword arguments are forwarded to `PromptingTools.aiembed`.

In practice you embed the chunks produced by the ingestion layer:

```julia
docs   = ingest(; sources = ["FunSQL.jl", "OMOP CDM"])
chunks = String[]
for d in docs
    for c in chunk_document(d)        # content-type-aware chunking
        push!(chunks, c.text)
    end
end

E = embed(chunks)                     # 768 × length(chunks)
```

## Validating embeddings

Before storing vectors, check that they are well-formed.
[`validate_embeddings`](@ref) runs a series of cheap structural checks and returns
a `(; dim, n, norms)` summary:

```julia
info = validate_embeddings(E; expected_dim = embedding_dimension(), chunks = chunks)
info.dim    # 768
info.n      # number of chunks
```

It throws on the first problem it finds:

1. an empty matrix (zero columns);
2. a dimension that does not match `expected_dim`;
3. a column count that does not match `length(chunks)`;
4. any non-finite entry (`NaN`/`Inf`);
5. a zero-vector (degenerate) column;
6. a column whose self-similarity is not `≈ 1`.

### Similarity sanity checks

Structural validity does not guarantee the model captures *meaning*. Two helpers
check semantics. [`cosine_similarity`](@ref) and [`similarity_matrix`](@ref) let
you inspect relationships directly:

```julia
cosine_similarity(view(E, :, 1), view(E, :, 2))   # similarity of the first two chunks
S = similarity_matrix(E)                           # full n × n cosine matrix (diagonal ≈ 1)
```

[`embedding_sanity_check`](@ref) is an end-to-end check that the model orders
meaning correctly — it embeds an anchor sentence, a semantically *near* sentence,
and an unrelated *far* one, and confirms `cos(anchor, near) > cos(anchor, far)`:

```julia
res = embedding_sanity_check()          # uses OMOP-flavoured defaults
res.passed                              # true when near beats far
res.near, res.far                       # the two similarity scores
```

This requires a reachable backend (it makes live embedding calls), so it belongs
in a smoke test rather than a unit test.

## Storing embeddings

All three backends share one interface, so you can swap where vectors live without
changing surrounding code:

| Store                    | Backing                          | Requirement                    |
|--------------------------|----------------------------------|--------------------------------|
| [`LocalVectorStore`](@ref) | in-memory matrix + `Serialization` | none — built in              |
| [`PgVectorStore`](@ref)    | PostgreSQL + `pgvector`          | `LibPQ` and a running server   |
| [`FaissVectorStore`](@ref) | a FAISS index                    | `Faiss.jl` loaded in `Main`    |

The common operations are [`add!`](@ref), [`search`](@ref), `length`, and — where
supported — [`save`](@ref) / [`load`](@ref).

### Local file-based store

The simplest option: vectors live in a `dim × n` matrix, search is cosine
similarity, and the whole store round-trips through `Serialization`. No external
service is required.

```julia
store = LocalVectorStore(embedding_dimension())    # 768-d store
add!(store, E, chunks)                              # append embeddings + their text
length(store)                                       # number of stored vectors

hits = search(store, embed("count patients"), 5)    # top-5 nearest
hits[1].chunk                                        # most similar chunk text
hits[1].score                                        # cosine similarity in [-1, 1]

save(store, "omop_index.jls")                        # persist
store2 = load(LocalVectorStore, "omop_index.jls")    # restore
```

Vectors are L2-normalised on insertion by default (`normalize=true`), which makes
cosine search a plain dot product and keeps scores in `[-1, 1]`.

### PostgreSQL + pgvector

For a shared, queryable index, back the store with PostgreSQL's `pgvector`
extension. [`PgVectorStore`](@ref) wraps
[`store_embeddings_pgvector`](@ref) and [`search_embeddings_pgvector`](@ref):

```julia
using LibPQ

conn  = LibPQ.Connection("postgresql://user:pass@localhost/health")
store = PgVectorStore(conn, embedding_dimension(); table = "omop_embeddings", metric = :cosine)

add!(store, E, chunks)                               # creates the table + inserts in one transaction
hits = search(store, embed("count patients"), 5)     # Vector{Hit}: row id in `index`, raw `distance` kept
```

`metric` chooses the distance operator: `:cosine` (`<=>`), `:dot` (`<#>`), or
`:l2` (`<->`). Table names are validated as SQL identifiers before interpolation.
The underlying functions can also be called directly if you do not want the store
wrapper:

```julia
store_embeddings_pgvector(conn, E, chunks, embedding_dimension(); table = "omop_embeddings")
hits = search_embeddings_pgvector(conn, embed("count patients"), 5; table = "omop_embeddings")  # raw (; id, chunk, distance)
```

!!! note "pgvector prerequisites"
    The target database must have the `vector` extension installed
    (`CREATE EXTENSION IF NOT EXISTS vector;`) and `LibPQ` available in the
    environment.

### FAISS

For large local indexes, [`FaissVectorStore`](@ref) delegates to
[FAISS](https://github.com/facebookresearch/faiss). FAISS is an **optional**
dependency: the store works only when a `Faiss` module is loaded into `Main`
(e.g. `using Faiss`), and otherwise raises a clear error at construction — nothing
else in the package requires it to be installed.

```julia
using Faiss                                          # optional; load before constructing

store = FaissVectorStore(embedding_dimension())      # inner-product index (cosine after normalisation)
add!(store, E, chunks)
hits = search(store, embed("count patients"), 5)     # Vector{Hit}, as with every backend
```

As with the local store, vectors are normalised for the inner-product/cosine
metrics; use `metric = :l2` for a raw L2 index (scores are then negated distances
so that larger is always more similar).

## Retrieving by text

[`search`](@ref) takes a query *vector*, so at query time you would normally embed
the question yourself and pass the result in. [`retrieve`](@ref) folds those two
steps into one per-query call: give it the store and a **string**, and it embeds
the query and returns the nearest chunks. It works with any store backend and
returns exactly what that backend's [`search`](@ref) yields.

```julia
store = LocalVectorStore(embedding_dimension())
add!(store, embed(chunks), chunks)

hits = retrieve(store, "How do I count patients in OMOP?", 5)
hits[1].chunk        # most relevant chunk text
```

`model`, `provider`, and any extra keyword arguments are forwarded to the embedder,
so you can retrieve with the same model you indexed with. The embedder itself is
injectable via the `embedder` keyword — handy for tests or a cached embedding
function — and is called as `embedder(query, model; provider, kwargs...)`.

## Putting it together

```julia
using HealthLLM

# 1. Gather and chunk documents.
docs   = ingest(; sources = ["FunSQL.jl", "OMOP CDM"])
chunks = [c.text for d in docs for c in chunk_document(d)]

# 2. Embed with the default dual-backend model.
E = embed(chunks)

# 3. Validate before storing.
validate_embeddings(E; expected_dim = embedding_dimension(), chunks = chunks)

# 4. Store locally (swap for PgVectorStore / FaissVectorStore as needed).
store = LocalVectorStore(embedding_dimension())
add!(store, E, chunks)
save(store, "omop_index.jls")

# 5. Query.
for h in search(store, embed("How do I count patients?"), 3)
    println(round(h.score, digits = 3), "  ", first(h.chunk, 80))
end
```

See the [API reference](index.md) for full docstrings of every function above.
```
