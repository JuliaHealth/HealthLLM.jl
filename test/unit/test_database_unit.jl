using Test
using HealthLLM

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "Database Module Unit Tests" begin
    @testset "_vector_to_pgarray formatting" begin
        # Float vector
        @test_valid ExactValidator() HealthLLM.Database._vector_to_pgarray([1.0, 2.0, 3.0]) "[1.0,2.0,3.0]"

        # Int vector
        @test_valid ExactValidator() HealthLLM.Database._vector_to_pgarray([1, 2, 3]) "[1,2,3]"

        # Empty vector
        @test_valid ExactValidator() HealthLLM.Database._vector_to_pgarray(Float64[]) "[]"

        # Higher dimension vector
        v = Float64[0.123456, -0.987654, 0.0]
        res = HealthLLM.Database._vector_to_pgarray(v)
        @test startswith(res, "[0.123456,")
        @test endswith(res, ",0.0]")
    end
end
