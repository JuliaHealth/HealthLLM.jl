module Query

using PromptingTools
using RAGTools

"""
    generate_funsql_query(index, model_embedding, model_name, prompt_template, question; schema=PromptingTools.OllamaSchema())

Run retrieval-augmented generation against `index` and return the generated answer.

# Arguments
- `index`: The RAG index to query against.
- `model_embedding::AbstractString`: Name of the embedding model for retrieving relevant chunks.
- `model_name::AbstractString`: Name of the chat model for generating the answer.
- `prompt_template::AbstractString`: Template string with `{input_query}` placeholder for the question.
- `question::AbstractString`: The user's question to answer.

# Keywords
- `schema=PromptingTools.OllamaSchema()`: The schema configuration for PromptingTools.

# Returns
The generated answer from the RAG query.

# Example

```julia
index = build_index_rag(files)
answer = generate_funsql_query(
    index,
    "nomic-embed-text",
    "llama3.2",
    "Context: {input_query}. Answer concisely.",
    "What is health?"
)
```
"""
function generate_funsql_query(
    index,
    model_embedding::AbstractString,
    model_name::AbstractString,
    prompt_template::AbstractString,
    question::AbstractString;
    schema=PromptingTools.OllamaSchema()
)

    prompt = replace(prompt_template, "{input_query}" => question)
    answer = RAGTools.airag(index;
        question=prompt,
        retriever_kwargs=(
            model=model_embedding,
            schema=schema,
            embedder_kwargs=(schema=schema, model=model_embedding)
        ),
        generator_kwargs=(model=model_name, schema=schema)
    )

    return answer
end

end