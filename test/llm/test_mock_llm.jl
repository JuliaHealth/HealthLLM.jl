using Test
using PromptingTools

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "Mock LLM Generation & Response Validation" begin
    # Configure deterministic mock LLM responses
    mock = MockLLM(Dict(
        "hypertension definition" => "Hypertension is defined as persistent elevated arterial blood pressure above 130/80 mmHg.",
        r"diabetes.*treatment" => "First line treatment for type 2 diabetes involves metformin and lifestyle changes.",
        "extract patient" => "```json\n{\"patient_id\": 101, \"condition\": \"hypertension\", \"systolic\": 142, \"confidence\": 0.95}\n```"
    ); default_response="Standard clinical response.")

    schema = MockHealthLLMSchema(mock)

    @testset "Text Generation with Normalized & RequiredContent Validators" begin
        # 1. Text output test
        msg = PromptingTools.aigenerate(schema, "Please provide the hypertension definition")
        response_text = msg.content

        # Normalized comparison
        @test_valid NormalizedTextValidator(lowercase=true, trim=true) response_text "hypertension is defined as persistent elevated arterial blood pressure above 130/80 mmhg."

        # Required content validation
        @test_valid RequiredContentValidator(
            required=["elevated", "blood pressure", "130/80"],
            forbidden=["cure guaranteed", "surgical emergency"],
            mode=:all
        ) response_text
    end

    @testset "Regex prompt matching" begin
        msg = PromptingTools.aigenerate(schema, "What is the recommended diabetes management and treatment plan?")
        @test_valid RequiredContentValidator(required=["metformin", "lifestyle"]) msg.content
    end

    @testset "Structured Output Validation" begin
        msg = PromptingTools.aigenerate(schema, "extract patient from note")
        json_response = msg.content

        jv = JSONStructureValidator(
            required_keys=["patient_id", "condition", "systolic", "confidence"],
            key_types=Dict("patient_id" => Integer, "condition" => AbstractString, "confidence" => Number),
            range_constraints=Dict("confidence" => (0.0, 1.0), "systolic" => (50.0, 250.0))
        )

        @test_valid jv json_response
    end

    @testset "Mock Embedding Generation" begin
        doc = "Clinical note on patient blood pressure"
        data_msg = PromptingTools.aiembed(schema, doc)
        @test data_msg.status == 200
        @test length(data_msg.content) == 8 # 8-dim mock vector
        @test isapprox(sum(data_msg.content .^ 2), 1.0; atol=0.01) # unit vector
    end
end
