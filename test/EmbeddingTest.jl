@testset "Embedding" begin
    using HealthLLM.Embedding
    using RAGTools

    @testset "build_index_rag" begin
        @testset "calls RAGTools.build_index with files" begin
            mock_files = ["file1.jl", "file2.jl"]
            mock_result = "mock_index"
            
            RAGTools.build_index_mock() do
                RAGTools.build_index(mock_files) |> returns(mock_result)
            end
        end
    end
end
