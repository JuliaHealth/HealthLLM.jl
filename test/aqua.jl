using Aqua
using HealthLLM

@testset "Aqua.jl quality checks" begin
    Aqua.test_unbound_args(HealthLLM)
    Aqua.test_undefined_exports(HealthLLM)
    Aqua.test_stale_deps(HealthLLM; ignore=[:HuggingFaceHub])
    Aqua.test_deps_compat(HealthLLM)
end
