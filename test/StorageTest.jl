using HealthLLM
using HealthLLM.Storage: LocalVectorStore, PgVectorStore, FaissVectorStore,
    add!, search, retrieve, save, load
using HealthLLM.Database: _pg_metric_op, _check_identifier, _pg_create_sql, _pg_search_sql

@testset "Storage" begin
    @testset "LocalVectorStore basics" begin
        store = LocalVectorStore(3)
        @test length(store) == 0

        E = Float32[1 0 0; 0 1 0; 0 0 1]     # 3 × 3 identity: three orthonormal vecs
        add!(store, E, ["x-axis", "y-axis", "z-axis"])
        @test length(store) == 3

        hits = search(store, Float32[0.9, 0.1, 0.0], 2)
        @test length(hits) == 2
        @test hits[1].chunk == "x-axis"
        @test hits[1].score > hits[2].score
        @test hits[1].score ≈ 1.0 atol = 0.2

        # normalization makes scores true cosine (bounded by 1)
        @test all(h -> h.score <= 1.0 + 1e-6, hits)
    end

    @testset "LocalVectorStore incremental add + k clamp" begin
        store = LocalVectorStore(2)
        add!(store, Float32[1 0; 0 1], ["a", "b"])
        add!(store, reshape(Float32[1, 1], 2, 1), ["c"])
        @test length(store) == 3
        @test length(search(store, Float32[1, 0], 10)) == 3   # k clamped to n
        @test isempty(search(LocalVectorStore(2), Float32[1, 0], 5))
    end

    @testset "LocalVectorStore validation" begin
        store = LocalVectorStore(4)
        @test_throws DimensionMismatch add!(store, rand(Float32, 3, 2), ["a", "b"])
        @test_throws DimensionMismatch add!(store, rand(Float32, 4, 2), ["only-one"])
        @test_throws DimensionMismatch search(store, rand(Float32, 3), 1)
        @test_throws ArgumentError search(store, rand(Float32, 4), 0)
        @test_throws ArgumentError LocalVectorStore(0)
    end

    @testset "LocalVectorStore save/load round-trip" begin
        store = LocalVectorStore(3)
        add!(store, Float32[1 0 0; 0 1 0; 0 0 1], ["x", "y", "z"])
        path = joinpath(mktempdir(), "store.jls")
        @test save(store, path) == path
        @test isfile(path)

        restored = load(LocalVectorStore, path)
        @test length(restored) == 3
        @test restored.dim == store.dim
        @test restored.chunks == store.chunks
        @test restored.embeddings == store.embeddings
        hits = search(restored, Float32[1, 0, 0], 1)
        @test hits[1].chunk == "x"
    end

    @testset "retrieve (text query via injected embedder)" begin
        store = LocalVectorStore(3)
        add!(store, Float32[1 0 0; 0 1 0; 0 0 1], ["x-axis", "y-axis", "z-axis"])

        # Deterministic stub embedder: maps the query text to an axis vector.
        # Signature must match how retrieve calls it: embedder(query, model; provider, kwargs...)
        function stub(text, model; provider=:ollama, kwargs...)
            v = startswith(text, "x") ? Float32[1, 0, 0] :
                startswith(text, "y") ? Float32[0, 1, 0] : Float32[0, 0, 1]
            return reshape(v, :, 1)
        end

        hits = retrieve(store, "x direction please", 2; embedder=stub)
        @test length(hits) == 2
        @test hits[1].chunk == "x-axis"
        @test hits[1].score > hits[2].score

        @test retrieve(store, "y things", 1; embedder=stub)[1].chunk == "y-axis"

        # Embedder may also return a bare vector (not a dim×1 matrix).
        vecstub(text, model; provider=:ollama, kwargs...) = Float32[0, 0, 1]
        @test retrieve(store, "anything", 1; embedder=vecstub)[1].chunk == "z-axis"

        @test_throws ArgumentError retrieve(store, ""; embedder=stub)
    end

    @testset "pgvector SQL helpers" begin
        @test _pg_metric_op(:cosine) == "<=>"
        @test _pg_metric_op(:dot) == "<#>"
        @test _pg_metric_op(:l2) == "<->"
        @test_throws ArgumentError _pg_metric_op(:manhattan)

        @test _check_identifier("embeddings") == "embeddings"
        @test _check_identifier("omop_chunks_2") == "omop_chunks_2"
        @test_throws ArgumentError _check_identifier("bad name")
        @test_throws ArgumentError _check_identifier("drop;table")
        @test_throws ArgumentError _check_identifier("2startswithdigit")

        create = _pg_create_sql("embeddings", 768)
        @test occursin("CREATE TABLE IF NOT EXISTS embeddings", create)
        @test occursin("VECTOR(768)", create)

        srch = _pg_search_sql("embeddings", :cosine)
        @test occursin("<=>", srch)
        @test occursin("ORDER BY", srch)
        @test occursin("LIMIT", srch)
    end

    @testset "PgVectorStore construction" begin
        # No live DB: verify field wiring and identifier validation only.
        store = PgVectorStore(nothing, 768; table="omop_embeddings", metric=:l2)
        @test store.dim == 768
        @test store.table == "omop_embeddings"
        @test store.metric == :l2
        @test_throws ArgumentError PgVectorStore(nothing, 0)
    end

    @testset "FaissVectorStore requires Faiss" begin
        # Faiss.jl is an optional dependency not loaded in the test env.
        @test !isdefined(Main, :Faiss)
        @test_throws ArgumentError FaissVectorStore(768)
    end
end
