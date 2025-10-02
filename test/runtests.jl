using DrWatson, Test
@quickactivate "HealthLLM"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "HealthLLM tests" begin
    include("FunSQLTest.jl")
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
