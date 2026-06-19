module Pipeline

using RAGTools
using PromptingTools
using ..Grounding
using ..Ingestion
using ..Query
using ..Utils

function setup_grounding_index(;
                               grounding_dir::Union{String,Nothing}=nothing,
                               embedder_model::Union{String,Nothing}=nothing,
                               verbose::Bool=true)
    index = Ingestion.build_grounding_index(; grounding_dir=grounding_dir, embedder_model=embedder_model, verbose=verbose)
    if verbose && RAGTools.chunks(index) !== nothing
        n = length(RAGTools.chunks(index))
        @info "Grounding index built with $n chunks"
    end
    index
end

function answer_question(
    question::String,
    index::RAGTools.AbstractDocumentIndex,
    model_embedding::String,
    model_name::String;
    template::Symbol=:FunSQLQueryGeneration,
    verbose::Bool=true,
    retriever_kwargs_overrides::NamedTuple=NamedTuple(),
    generator_kwargs_overrides::NamedTuple=NamedTuple())
    result = Query.generate_funsql_query(
        index, model_embedding, model_name, question;
        template=template,
        retriever_kwags_overrides=retriever_kwargs_overrides,
        generator_kwargs_overrides=generator_kwargs_overrides)
    if verbose
        println("--- Generated FunSQL ---")
        println(result)
        println("--- Sources ---")
        for s in result.sources
            println("  - $s")
        end
    end
    return result
end

function answer_question_direct(question::String,
                                model_embedding::String,
                                model_name::String;
                                template::Symbol=:FunSQLQueryDirect,
                                verbose::Bool=true,
                                generator_kwargs_overrides::NamedTuple=NamedTuple())
    result = Query.generate_funsql_query_direct(
        model_embedding, model_name, question;
        template=template,
        generator_kwargs_overrides=generator_kwargs_overrides)
    if verbose
        println("--- Generated FunSQL (no RAG) ---")
        println(result)
    end
    return result
end

function quick_demo(; model_embedding="hf:sentence-transformers/all-mpnet-base-v2",
                      model_name="hf:Qwen/Qwen2.5-Coder-1.5B-Instruct")
    gen_schema = Utils.get_schema(nothing, model_name)
    emb_schema = Utils.get_schema(nothing, model_embedding)
    PromptingTools.register_model!(name=model_name, schema=gen_schema)
    PromptingTools.register_model!(name=model_embedding, schema=emb_schema)
    PromptingTools.MODEL_CHAT = model_name
    PromptingTools.MODEL_EMBEDDING = model_embedding

    index = setup_grounding_index(; embedder_model=model_embedding)

    questions = [
        "Count patients per care site place of service",
        "Find the minimum, maximum and average length of condition episodes",
        "List patient counts by age and gender for patients with a specific condition",
    ]
    for q in questions
        println("\n" * "^" * 60)
        println("Q: $q")
        println("---")
        answer = answer_question(q, index, model_embedding, model_name)
        println("^" * 60)
    end
    index
end

end
