module Query

using PromptingTools
using RAGTools

function generate_funsql_query(index, model_embedding::String, model_name::String, prompt_template::String, question::String)
    prompt = replace(prompt_template, "{input_query}" => question)

    # infer appropriate schemas for generator and retriever (supports HuggingFace, Ollama, etc.)
    gen_schema = Utils.get_schema(nothing, model_name)
    emb_schema = Utils.get_schema(nothing, model_embedding)

    answer = RAGTools.airag(index;
        question=prompt,
        retriever_kwargs=(model=model_embedding, schema=emb_schema, embedder_kwargs=(schema=emb_schema, model=model_embedding)),
        generator_kwargs=(model=model_name, schema=gen_schema)
    )

    return answer
end

end
