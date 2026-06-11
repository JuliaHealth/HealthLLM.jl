module Query

using PromptingTools
using RAGTools
using ..Grounding

function generate_funsql_query(index,
                               model_embedding::String,
                               model_name::String,
                               question::String;
                               template::Symbol=:FunSQLQueryGeneration,
                               retriever_kwags_overrides::NamedTuple=NamedTuple(),
                               generator_kwargs_overrides::NamedTuple=NamedTuple())
    gen_schema = Utils.get_schema(nothing, model_name)
    emb_schema = Utils.get_schema(nothing, model_embedding)

    Grounding.register_funsql_template!(name=template)

    retriever_kwargs = (model=model_embedding, schema=emb_schema, embedder_kwargs=(schema=emb_schema, model=model_embedding))
    generator_kwargs = (model=model_name, schema=gen_schema, template=template)

    retriever_kwargs = merge(retriever_kwargs, retriever_kwags_overrides)
    generator_kwargs = merge(generator_kwargs, generator_kwargs_overrides)

    answer = RAGTools.airag(index;
        question=question,
        retriever_kwargs=retriever_kwargs,
        generator_kwargs=generator_kwargs)

    return answer
end

function generate_funsql_query_direct(model_embedding::String,
                                      model_name::String,
                                      question::String;
                                      template::Symbol=:FunSQLQueryDirect,
                                      generator_kwargs_overrides::NamedTuple=NamedTuple())
    gen_schema = Utils.get_schema(nothing, model_name)

    Grounding.register_funsql_template_no_context!(name=template)

    generator_kwargs = (model=model_name, schema=gen_schema, template=template)
    generator_kwargs = merge(generator_kwargs, generator_kwargs_overrides)

    msg = PromptingTools.aigenerate(question; generator_kwargs...)
    return msg.content
end

end
