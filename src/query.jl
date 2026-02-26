module Query

using PromptingTools
using RAGTools

"""
    generate_funsql_query(index, model_embedding, model_name, prompt_template, question; schema=PromptingTools.OllamaSchema())

Run retrieval-augmented generation against `index` and return the generated answer.
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
