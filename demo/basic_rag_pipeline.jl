using DrWatson
@quickactivate "HealthLLM"
using HealthLLM
using PromptingTools
using RAGTools

println("=" ^ 60)
println("HealthLLM.jl — Baseline RAG Pipeline Demo")
println("=" ^ 60)

model_embedding = "hf:sentence-transformers/all-mpnet-base-v2"
model_name = "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct"

println("\n[1/4] Registering models and schemas...")
emb_schema = HealthLLM.Utils.get_schema(nothing, model_embedding)
gen_schema = HealthLLM.Utils.get_schema(nothing, model_name)
println("  Embedding schema: ", typeof(emb_schema))
println("  Generator schema: ", typeof(gen_schema))

PromptingTools.register_model!(name=model_name, schema=gen_schema)
PromptingTools.register_model!(name=model_embedding, schema=emb_schema)
PromptingTools.MODEL_CHAT = model_name
PromptingTools.MODEL_EMBEDDING = model_embedding

println("\n[2/4] Building grounding index from curated docs...")
index = HealthLLM.build_grounding_index(; embedder_model=model_embedding, verbose=true)
n_chunks = length(RAGTools.get_chunks(index))
println("  $n_chunks chunks indexed from grounding corpus.")

println("\n[3/4] Registering FunSQL prompt template...")
HealthLLM.register_funsql_template!()
println("  Template :FunSQLQueryGeneration registered.")

questions = [
    "Count patients per care site place of service",
    "List patient counts by age and gender for patients with hip fracture",
]

println("\n[4/4] Running RAG queries...")
for (i, q) in enumerate(questions)
    println("\n" * "-" * 60)
    println("Query $i: $q")
    println("-" * 60)
    try
        result = HealthLLM.answer_question(q, index, model_embedding, model_name; verbose=true)
        answer_text = result isa RAGTools.RAGResult ? result.final_answer : string(result)
        println("\nFinal answer:\n$answer_text")
    catch e
        println("Error during query: $e")
        if e isa MethodError
            showerror(stdout, e, catch_backtrace())
        end
    end
end

println("\n" * "=" * 60)
println("Demo complete.")
println("=" * 60)
