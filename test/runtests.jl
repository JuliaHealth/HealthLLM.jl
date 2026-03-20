using Test
using Pkg: Pkg

@testset "Pgvector" begin
    include("PgvectorTest.jl")
end

@testset "Utils" begin
    include("UtilsTest.jl")
end

@testset "Database" begin
    include("DatabaseTest.jl")
end

@testset "Embedding" begin
    include("EmbeddingTest.jl")
end

@testset "Query" begin
    include("QueryTest.jl")
end

@testset "FunSQL" begin
    include("FunSQLTest.jl")
end