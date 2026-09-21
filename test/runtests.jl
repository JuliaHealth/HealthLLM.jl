using Test

println("Starting tests")
ti = time()

@testset "HealthLLM tests" begin
    include("ExportsTest.jl")
    include("HuggingFaceTest.jl")
    include("UtilsTest.jl")
    include("DatabaseTest.jl")
    include("QueryTest.jl")
    include("FunSQLTest.jl")
    include("IngestionTest.jl")
    include("EmbeddingsTest.jl")
    include("StorageTest.jl")
    include("PromptTest.jl")
    include("ExecutionTest.jl")
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
