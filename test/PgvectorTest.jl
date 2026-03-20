@testset "Pgvector" begin
    using HealthLLM.Pgvector

    @testset "to_pgvector_literal" begin
        @test to_pgvector_literal([1, 2, 3]) == "[1,2,3]"
        @test to_pgvector_literal([1.5, 2.5, 3.5]) == "[1.5,2.5,3.5]"
        @test to_pgvector_literal(Float32[0.1, 0.2]) == "[0.1,0.2]"
        @test to_pgvector_literal(Int[] ) == "[]"
    end
end
