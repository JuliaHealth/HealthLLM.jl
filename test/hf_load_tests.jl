using Test
using HealthLLM

@testset "HuggingFace load helper" begin
    # Case: HuggingFaceHub not available in this test environment
    res = HealthLLM.load_huggingface_model("some/fake-model-name")
    @test isa(res, HealthLLM.HuggingFaceLoadResult)
    @test res.downloaded == false
    @test (res.path === nothing)
    @test res.info == "some/fake-model-name"

    # If HuggingFaceHub is available, we do a light check — don't require network
    if isdefined(Main, :HuggingFaceHub)
        HF = Main.HuggingFaceHub
        try
            # Try to get model info; some HF installs provide offline metadata
            info = HF.info(HF.Model, "gpt2")
            # Call the helper and ensure we get a HuggingFaceLoadResult
            res2 = HealthLLM.load_huggingface_model("gpt2")
            @test isa(res2, HealthLLM.HuggingFaceLoadResult)
            @test (res2.downloaded == true || res2.info !== nothing)
        catch err
            @info "HuggingFaceHub present but info query failed: $err"
        end
    end
end
