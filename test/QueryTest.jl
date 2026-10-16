using HealthLLM
using RAGTools

@testset "Query" begin
    @testset "generate_funsql_query template replacement" begin
        template = "Context: {input_query}. Answer the question."
        question = "What is health?"
        expected = "Context: What is health?. Answer the question."
        prompt = replace(template, HealthLLM.Query.QUERY_PLACEHOLDER => question)
        @test prompt == expected
    end

    @testset "default template shares the FunSQL grounding rules" begin
        # One source of truth for grounding across the airag path and build_prompt.
        @test occursin(HealthLLM.FUNSQL_SYSTEM_PROMPT, HealthLLM.DEFAULT_QUERY_TEMPLATE)
        @test occursin(HealthLLM.Query.QUERY_PLACEHOLDER, HealthLLM.DEFAULT_QUERY_TEMPLATE)
    end

    @testset "question-only form accepts a schema override" begin
        # `schema` used to be accepted and silently ignored; it must now reach
        # airag, so a bogus schema has to surface as an error rather than pass.
        @test_throws Exception HealthLLM.generate_funsql_query(
            RAGTools.SimpleIndexer(), "nomic-embed-text", "llama3.2",
            "What is health?"; schema=:not_a_schema
        )
    end

    @testset "generate_funsql_query dispatches correctly" begin
        index = RAGTools.SimpleIndexer()
        @test_throws Exception HealthLLM.generate_funsql_query(
            index,
            "nomic-embed-text",
            "llama3.2",
            "Context: {input_query}. Answer concisely.",
            "What is health?"
        )
    end
end
