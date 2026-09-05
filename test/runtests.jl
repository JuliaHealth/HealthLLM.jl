using Test

println("Starting tests")
ti = time()

@testset "HealthLLM tests" begin
    include("UtilsTest.jl")
    include("DatabaseTest.jl")
    include("QueryTest.jl")
    include("utils_tests.jl")
    include("hf_model_test.jl")
    include("hf_load_tests.jl")
    include("FunSQLTest.jl")
    include("IngestionTest.jl")
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
