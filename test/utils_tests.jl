using Test
using DrWatson
@quickactivate "HealthLLM"

using HealthLLM

@testset "Utils helpers" begin
    # Progress callback registration and reporting
    called = false
    vals = nothing
    HealthLLM.Utils.register_progress!( (c,t,m) -> (called = true; vals = (c,t,m)) )
    HealthLLM.Utils.report_progress(2, 5; msg="hello")
    @test called == true
    @test vals == (2, 5, "hello")

    # Clearing should not error and should replace the callback
    HealthLLM.Utils.clear_progress!()
    # This should call the no-op callback (no error expected)
    HealthLLM.Utils.report_progress(1, 1; msg="x")

    # Explicit schema ctor lookup: inject a fake schema type into PromptingTools
    @eval PromptingTools begin
        struct FakeSchema end
    end
    sch = HealthLLM.Utils.get_schema("Fake", nothing)
    @test typeof(sch) == PromptingTools.FakeSchema

    # collect_files_with_extensions + write_combined_file
    tmpdir = mktempdir()
    f1 = joinpath(tmpdir, "a.txt"); open(f1, "w") do io write(io, "A") end
    f2 = joinpath(tmpdir, "b.jl"); open(f2, "w") do io write(io, "B") end
    nested = joinpath(tmpdir, "sub"); mkpath(nested)
    f3 = joinpath(nested, "c.txt"); open(f3, "w") do io write(io, "C") end

    files = HealthLLM.Utils.collect_files_with_extensions(tmpdir, [".txt"]) |> sort
    @test files == sort([f1, f3])

    out = joinpath(tmpdir, "combined.txt")
    HealthLLM.Utils.write_combined_file([f1, f2], out)
    s = read(out, String)
    @test occursin("# File: $f1", s) && occursin("# File: $f2", s)

    # Simulate HuggingFaceHub availability by injecting a mock into Main
    prevDefined = isdefined(Main, :HuggingFaceHub)
    prevHF = prevDefined ? Main.HuggingFaceHub : nothing

    # Create a module at top-level and set a global HuggingFaceHub binding
    modcode = "module MockHF; export snapshot; snapshot(x) = \"/tmp/mocked\"; end; global HuggingFaceHub = MockHF"
    Base.include_string(Main, modcode)
    res = HealthLLM.Utils.load_huggingface_model("fake-model")
    @test isa(res, HealthLLM.Utils.HuggingFaceLoadResult)
    @test res.downloaded == true
    @test res.path == "/tmp/mocked"

    # Restore previous state
    if prevDefined
        Main.HuggingFaceHub = prevHF
    else
        # remove our MockHF and HuggingFaceHub bindings
        Base.delete_binding(Main, :HuggingFaceHub)
        Base.delete_binding(Main, :MockHF)
    end
end
