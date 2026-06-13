using Aqua
using HealthLLM

@testset "Aqua.jl quality checks" begin
    Aqua.test_all(HealthLLM; ambiguities=true, stale_deps=true, deps_compat=true)
end
