using Test
using HealthLLM
using PromptingTools

if !@isdefined(TestHarness)
    include(normpath(joinpath(@__DIR__, "..", "harness", "TestHarness.jl")))
    using .TestHarness
end

@testset "End-to-End Synthetic RAG Pipeline Integration" begin
    mktempdir() do tmpdir
        # 1. Prepare synthetic medical corpus
        corpus_files = create_test_corpus(tmpdir)
        @test length(corpus_files) == 3

        # 2. File collection
        collected = HealthLLM.collect_files_with_extensions(tmpdir, [".txt", ".md"])
        @test length(collected) == 3

        # 3. Consolidate into reference document
        combined_file = joinpath(tmpdir, "clinical_reference.txt")
        HealthLLM.write_combined_file(collected, combined_file)
        @test isfile(combined_file)

        # 4. Read combined text & create chunks
        raw_text = read(combined_file, String)
        chunks = [strip(c) for c in split(raw_text, "# File:") if !isempty(strip(c))]
        @test length(chunks) >= 3

        # 5. Build mock RAG index
        mock_index = create_mock_rag_index(chunks)

        # 6. Mock LLM for clinical QA
        mock = MockLLM(Dict(
            "hypertension" => "Based on the clinical notes, the patient is on lisinopril 10mg daily for stage 2 hypertension.",
            "diabetes" => "First line therapy for Type 2 Diabetes is metformin.",
            "cad" => "Target LDL cholesterol level for CAD patients is < 70 mg/dL."
        ))
        schema = MockHealthLLMSchema(mock)

        # 7. Query pipeline & Validation
        msg1 = PromptingTools.aigenerate(schema, "What medication is prescribed for hypertension?")
        @test_valid RequiredContentValidator(required=["lisinopril", "stage 2 hypertension"]) msg1.content

        msg2 = PromptingTools.aigenerate(schema, "What is the LDL target for cad?")
        @test_valid RequiredContentValidator(required=["LDL", "70 mg/dL"]) msg2.content
    end
end
