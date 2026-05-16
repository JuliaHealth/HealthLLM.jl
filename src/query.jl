module Query

using PromptingTools
using RAGTools
using LibPQ

function retrieve_similar(conn::LibPQ.Connection, query_embedding::AbstractVector, top_k::Int=5)
    embedding_str = string("[", join(query_embedding, ","), "]")
    result = LibPQ.execute(conn, """
        SELECT chunk, 1 - (embedding <=> \$1) AS similarity
        FROM embeddings
        ORDER BY embedding <=> \$1
        LIMIT \$2
    """, (embedding_str, top_k))
    
    chunks = []
    similarities = []
    for row in result
        push!(chunks, row["chunk"])
        push!(similarities, row["similarity"])
    end
    return chunks, similarities
end

function generate_funsql_query(index, model_embedding::String, model_name::String, prompt_template::String, question::String)
    prompt = replace(prompt_template, "{input_query}" => question)

    answer = RAGTools.airag(index; 
        question=prompt,
        retriever_kwargs=(model=model_embedding, schema=PromptingTools.OllamaSchema(), embedder_kwargs=(schema=PromptingTools.OllamaSchema(), model=model_embedding)),
        generator_kwargs=(model=model_name, schema=PromptingTools.OllamaSchema())
    )

    return answer
end

function generate_funsql_query_db(conn::LibPQ.Connection, embedder, model_name::String, prompt_template::String, question::String)
    # Embed the question
    query_embedding = PromptingTools.embed_question(embedder, question)
    
    # Retrieve similar chunks
    chunks, similarities = retrieve_similar(conn, query_embedding)
    
    # Use the retrieved chunks as context
    context = join(chunks, "\n")
    prompt = replace(prompt_template, "{input_query}" => question)
    prompt = replace(prompt, "{context}" => context)
    
    # Generate answer
    answer = PromptingTools.aigenerate(prompt; model=model_name, schema=PromptingTools.OllamaSchema())
    
    return answer
end

end
