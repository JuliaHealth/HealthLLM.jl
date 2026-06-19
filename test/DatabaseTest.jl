using HealthLLM
using HealthLLM.Database: validate_embeddings_inputs

@testset "Database" begin
    @testset "_vector_to_pgarray" begin
        @test HealthLLM.Database._vector_to_pgarray([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"
        @test HealthLLM.Database._vector_to_pgarray([1, 2, 3]) == "[1,2,3]"
        @test HealthLLM.Database._vector_to_pgarray(Float64[]) == "[]"
        @test HealthLLM.Database._vector_to_pgarray(Float32[0.1, 0.2]) == "[0.1,0.2]"
    end

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
