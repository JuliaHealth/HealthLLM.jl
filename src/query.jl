module Query

using PromptingTools
using RAGTools
using ..Providers

function generate_funsql_query(index, embedder::EmbeddingProvider, generator::ModelProvider, prompt_template::String, question::String)
    prompt = replace(prompt_template, "{input_query}" => question)

    # Since RAGTools.airag expects specific kwargs, we need to adapt
    # For now, assume index is built with the embedder, and use the generator for final answer

    # Retrieve relevant chunks
    question_emb = embed(embedder, [question])[1]
    scores = index.embeddings' * question_emb
    idxs = sortperm(scores; rev=true)[1:min(5, length(scores))]
    chunks = [index.chunks[i] for i in idxs]

    context = join(chunks, "\n\n")

    full_prompt = replace(prompt_template, "{input_query}" => question) * "\n\nContext:\n" * context

    answer = generate(generator, full_prompt)

    return answer
end

# Backward compatibility
function generate_funsql_query(index, model_embedding::String, model_name::String, prompt_template::String, question::String)
    # Create providers, assuming Ollama
    embedder = HuggingFaceEmbedder()  # Or something
    generator = OllamaProvider(model=model_name)
    generate_funsql_query(index, embedder, generator, prompt_template, question)
end

end