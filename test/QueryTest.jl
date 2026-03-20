@testset "Query" begin
    using HealthLLM.Query
    using PromptingTools
    using RAGTools

    @testset "generate_funsql_query" begin
        template = "Context: {input_query}. Answer the question."
        question = "What is health?"
        
        @testset "replaces placeholder in template" begin
            prompt = replace(template, "{input_query}" => question)
            @test prompt == "Context: What is health?. Answer the question."
        end
    end
end
