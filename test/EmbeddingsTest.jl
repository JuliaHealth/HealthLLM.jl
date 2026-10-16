using HealthLLM
using HealthLLM.Embeddings: EmbeddingModel, EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL,
    embedding_model, embedding_ref, embedding_dimension,
    cosine_similarity, similarity_matrix, validate_embeddings
using LinearAlgebra

@testset "Embeddings" begin
    @testset "registry" begin
        @test haskey(EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL)
        @test DEFAULT_EMBEDDING_MODEL == "nomic-embed-text"
        for (name, m) in EMBEDDING_MODELS
            @test m isa EmbeddingModel
            @test m.name == name
            @test !isempty(m.ollama)
            @test occursin("/", m.huggingface)   # HF repos are "org/name"
            @test m.dim > 0
        end
        @test embedding_dimension("nomic-embed-text") == 768
        @test embedding_dimension("all-minilm") == 384
    end

    @testset "lookup errors" begin
        @test embedding_model("bge-m3").dim == 1024
        @test_throws ArgumentError embedding_model("does-not-exist")
    end

    @testset "embedding_ref" begin
        @test embedding_ref("nomic-embed-text"; provider=:ollama) == "nomic-embed-text"
        @test embedding_ref("all-minilm"; provider=:huggingface) ==
            "hf:sentence-transformers/all-MiniLM-L6-v2"
        @test_throws ArgumentError embedding_ref("nomic-embed-text"; provider=:nonsense)
    end

    @testset "cosine_similarity" begin
        @test cosine_similarity([1.0, 0.0], [1.0, 0.0]) ≈ 1.0
        @test cosine_similarity([1.0, 0.0], [0.0, 1.0]) ≈ 0.0 atol = 1e-12
        @test cosine_similarity([1.0, 0.0], [-1.0, 0.0]) ≈ -1.0
        @test cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0   # zero vector guard
        @test_throws DimensionMismatch cosine_similarity([1.0], [1.0, 2.0])
    end

    @testset "similarity_matrix" begin
        E = Float32[1 0 1; 0 1 1]   # columns: e1, e2, e1+e2
        S = similarity_matrix(E)
        @test size(S) == (3, 3)
        @test all(isapprox.(diag(S), 1.0; atol=1e-6))
        @test S ≈ S'                            # symmetric
        @test S[1, 2] ≈ 0.0 atol = 1e-6         # orthogonal columns
        @test S[1, 3] ≈ 1 / sqrt(2) atol = 1e-6
    end

    @testset "validate_embeddings" begin
        E = Float32[1 0 1; 0 1 1]
        info = validate_embeddings(E; expected_dim=2, chunks=["a", "b", "c"])
        @test info.dim == 2
        @test info.n == 3
        @test length(info.norms) == 3

        @test_throws ArgumentError validate_embeddings(Matrix{Float32}(undef, 4, 0))
        @test_throws DimensionMismatch validate_embeddings(E; expected_dim=99)
        @test_throws DimensionMismatch validate_embeddings(E; chunks=["only-one"])

        bad = copy(E); bad[1, 1] = NaN
        @test_throws ArgumentError validate_embeddings(bad)

        zerocol = Float32[1 0; 0 0]
        @test_throws ArgumentError validate_embeddings(zerocol)
    end
end
