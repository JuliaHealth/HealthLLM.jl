using Test
using HealthLLM

@testset "Utils tests" begin
    @testset "_vector_to_pgarray" begin
        @test HealthLLM.Database._vector_to_pgarray([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"
        @test HealthLLM.Database._vector_to_pgarray([1, 2, 3]) == "[1,2,3]"
        @test HealthLLM.Database._vector_to_pgarray(Float64[]) == "[]"
    end

    @testset "build_index_rag dispatches correctly" begin
        @test_throws Exception HealthLLM.build_index_rag(
            RAGTools.SimpleIndexer(), ["Hello world"]; embedder_kwargs=(;)
        )
    end
end
