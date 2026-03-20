@testset "Database" begin
    using HealthLLM.Database

    @testset "validate_embeddings_inputs" begin
        embeddings = rand(384, 10)
        chunks = ["chunk $i" for i in 1:10]

        @test validate_embeddings_inputs(embeddings, chunks, 384) === nothing

        @test_throws DimensionMismatch validate_embeddings_inputs(
            rand(128, 10), chunks, 384
        )
        @test_throws DimensionMismatch validate_embeddings_inputs(
            embeddings, chunks[1:5], 384
        )
    end
end
