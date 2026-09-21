"""
    Query

End-to-end retrieval-augmented querying via `RAGTools.airag`: retrieval,
generation, and the prompt wrapper that carries the FunSQL grounding rules into
the question.

Unlike the `Prompt`/`Execution` modules — which format retrieved chunks
themselves and call the model directly — this module hands the whole loop to
`RAGTools.airag` and only decorates the question. Both paths share
[`FUNSQL_SYSTEM_PROMPT`](@ref) as their single source of grounding rules.
"""
module Query

using PromptingTools
using RAGTools
using ..Utils: get_schema
using ..Prompt: FUNSQL_SYSTEM_PROMPT

export generate_funsql_query, DEFAULT_QUERY_TEMPLATE, QUERY_PLACEHOLDER

"""
    QUERY_PLACEHOLDER

The substring a query template substitutes the user's question into: `"{input_query}"`.
"""
const QUERY_PLACEHOLDER = "{input_query}"

"""
    DEFAULT_QUERY_TEMPLATE

Default template for [`generate_funsql_query`](@ref): the shared
[`FUNSQL_SYSTEM_PROMPT`](@ref) grounding rules followed by the question slot.
Using the same system prompt as `build_prompt` keeps the two generation
paths from drifting apart in what they tell the model.
"""
const DEFAULT_QUERY_TEMPLATE = string(
    FUNSQL_SYSTEM_PROMPT, "\n\n# Analytical question\n", QUERY_PLACEHOLDER,
    "\n\n# FunSQL query")

"""
    generate_funsql_query(index, model_embedding, model_name, question;
                          template=DEFAULT_QUERY_TEMPLATE, schema=nothing, kwargs...)
    generate_funsql_query(index, model_embedding, model_name, template, question; kwargs...)

Run retrieval-augmented generation against `index` and return the generated answer.

The question is substituted into `template` at [`QUERY_PLACEHOLDER`](@ref)
(`"{input_query}"`) and the result is handed to `RAGTools.airag`, which does its
own retrieval and generation.

# Arguments
- `index`: The RAG index to query against.
- `model_embedding::AbstractString`: Embedding model used to retrieve relevant chunks.
- `model_name::AbstractString`: Chat model used to generate the answer.
- `question::AbstractString`: The user's question.

# Keywords
- `template::AbstractString=DEFAULT_QUERY_TEMPLATE`: Wrapper containing
  `"{input_query}"`. Defaults to the shared FunSQL grounding rules.
- `schema=nothing`: PromptingTools schema for both the generator and the embedder.
  When `nothing` (the default) a schema is inferred from each model name.
- `kwargs...`: Forwarded to `RAGTools.airag`.

# Returns
The generated answer from the RAG query.

# Example

```julia
index  = build_index_rag(RAGTools.SimpleIndexer(), files)
answer = generate_funsql_query(index, "nomic-embed-text", "llama3.2",
                               "How many distinct patients had a diabetes diagnosis?")
```
"""
function generate_funsql_query(
    index,
    model_embedding::AbstractString,
    model_name::AbstractString,
    question::AbstractString;
    template::AbstractString=DEFAULT_QUERY_TEMPLATE,
    schema=nothing,
    kwargs...
)
    prompt = replace(template, QUERY_PLACEHOLDER => question)
    gen_schema = schema === nothing ? get_schema(nothing, String(model_name)) : schema
    emb_schema = schema === nothing ? get_schema(nothing, String(model_embedding)) : schema

    return RAGTools.airag(index;
        question=prompt,
        retriever_kwargs=(model=model_embedding, schema=emb_schema,
            embedder_kwargs=(schema=emb_schema, model=model_embedding)),
        generator_kwargs=(model=model_name, schema=gen_schema),
        kwargs...)
end

# Positional-template form, kept for callers that supply their own wrapper.
generate_funsql_query(index, model_embedding::AbstractString, model_name::AbstractString,
    template::AbstractString, question::AbstractString; kwargs...) =
    generate_funsql_query(index, model_embedding, model_name, question;
        template=template, kwargs...)

end
