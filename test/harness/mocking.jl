# Mock LLM & Offline Schema Support

using PromptingTools
using RAGTools
using SparseArrays
using LinearAlgebra

"""
    MockLLM

A deterministic mock LLM for testing. Maps prompt strings or regular expressions to canned responses.

# Fields
- `responses::Dict{Any, String}`: Dict of (string/regex => response).
- `history::Vector{String}`: Record of all prompts received by this mock.
- `default_response::String`: Fallback response if no pattern matches.

# Examples
```julia
mock = MockLLM(Dict(
    "hypertension" => "Hypertension is high blood pressure.",
    r"diabetes.*treatment" => "Common treatments include insulin and metformin."
))
mock("Tell me about hypertension") # "Hypertension is high blood pressure."
```
"""
mutable struct MockLLM
    responses::Dict{Any, String}
    history::Vector{String}
    default_response::String

    function MockLLM(responses::Dict=Dict(); default_response="Mocked LLM Response")
        new(responses, String[], default_response)
    end
end

function (m::MockLLM)(prompt::AbstractString)
    push!(m.history, String(prompt))
    for (pat, resp) in m.responses
        if pat isa Regex
            if occursin(pat, prompt)
                return resp
            end
        elseif pat isa AbstractString
            if occursin(lowercase(String(pat)), lowercase(prompt))
                return resp
            end
        end
    end
    return m.default_response
end

"""
    MockHealthLLMSchema <: PromptingTools.AbstractPromptSchema

Custom schema for `PromptingTools` that bypasses all network calls and returns deterministic mock responses.
"""
struct MockHealthLLMSchema <: PromptingTools.AbstractPromptSchema
    mock::MockLLM
end

MockHealthLLMSchema() = MockHealthLLMSchema(MockLLM())

# Extend PromptingTools.aigenerate for MockHealthLLMSchema
function PromptingTools.aigenerate(
    schema::MockHealthLLMSchema,
    prompt;
    model::String="mock-model",
    kwargs...
)
    # Extract string from prompt (which may be a string, AITemplate, or vector of messages)
    prompt_str = if prompt isa AbstractString
        String(prompt)
    elseif prompt isa AbstractVector
        join([get(m, :content, string(m)) for m in prompt], "\n")
    else
        string(prompt)
    end

    response_text = schema.mock(prompt_str)

    # Return an AIMessage compatible with PromptingTools
    return PromptingTools.AIMessage(
        content=response_text,
        status=200,
        tokens=(10, 10),
        elapsed=0.01
    )
end

# Extend PromptingTools.aiembed for MockHealthLLMSchema
function PromptingTools.aiembed(
    schema::MockHealthLLMSchema,
    doc;
    model::String="mock-embedder",
    kwargs...
)
    # Generate deterministic pseudo-embedding vector based on doc hash
    h = hash(string(doc))
    # 8-dimensional mock embedding vector
    v = Float32[sin(h + i) for i in 1:8]
    v = v ./ norm(v)
    return PromptingTools.DataMessage(
        content=v,
        status=200,
        tokens=(5, 0),
        elapsed=0.005
    )
end

"""
    create_mock_rag_index(chunks::AbstractVector{<:AbstractString}=["Hypertension is high blood pressure.", "Diabetes affects insulin regulation."])

Builds a lightweight in-memory RAG index for testing `HealthLLM.generate_funsql_query` and RAG pipelines without external models.
"""
function create_mock_rag_index(chunks::AbstractVector{<:AbstractString}=["Hypertension is high blood pressure.", "Diabetes affects insulin regulation."])
    chunks_str = String[String(c) for c in chunks]
    n = length(chunks_str)
    dim = 8
    # Deterministic embeddings matrix (dim x n)
    emb_matrix = zeros(Float32, dim, n)
    for i in 1:n
        h = hash(chunks_str[i])
        v = Float32[sin(h + j) for j in 1:dim]
        emb_matrix[:, i] .= v ./ norm(v)
    end

    # Return a ChunkIndex / Mock index compatible with RAGTools
    try
        if isdefined(RAGTools, :ChunkIndex)
            return RAGTools.ChunkIndex(;
                id="mock-health-index",
                chunks=chunks_str,
                embeddings=emb_matrix,
                sources=fill("mock_doc.txt", n)
            )
        end
    catch _
    end

    return (chunks=chunks_str, embeddings=emb_matrix, sources=fill("mock_doc.txt", n))
end
