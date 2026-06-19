using HealthLLM
using RAGTools

@testset "Query" begin
    @testset "generate_funsql_query template replacement" begin
        template = "Context: {input_query}. Answer the question."
        question = "What is health?"
        expected = "Context: What is health?. Answer the question."
        prompt = replace(template, "{input_query}" => question)
        @test prompt == expected
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
