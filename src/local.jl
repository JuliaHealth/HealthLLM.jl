module LocalLLM

using PromptingTools
using PromptingTools: AbstractPromptSchema, DataMessage
import PromptingTools: aiembed, aigenerate
using PythonCall
using LinearAlgebra

export LocalManagedSchema

const DEFAULT_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"
const DEFAULT_GENERATION = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

struct LocalManagedSchema <: AbstractPromptSchema end

const _PY_MODULES = Dict{Symbol, Any}()
const _MODELS = Dict{String, Any}()

_py(mod::AbstractString) = get!(_PY_MODULES, Symbol(mod)) do
    pyimport(mod)
end

function _strip_prefix(model::AbstractString)
    model = replace(model, r"^local:" => "")
    model = replace(model, r"^hf:" => "")
    return model
end

function load_embedding_model(model::AbstractString=DEFAULT_EMBEDDING)
    model = _strip_prefix(model)
    key = "emb:$model"
    haskey(_MODELS, key) && return _MODELS[key]
    st = _py("sentence_transformers")
    @info "Downloading/loading embedding model: $model"
    m = st.SentenceTransformer(model)
    _MODELS[key] = m
    return m
end

function load_generation_model(model::AbstractString=DEFAULT_GENERATION)
    model = _strip_prefix(model)
    key = "gen:$model"
    haskey(_MODELS, key) && return _MODELS[key]
    transformers = _py("transformers")
    @info "Downloading/loading generation model: $model"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model_obj = transformers.AutoModelForCausalLM.from_pretrained(model)
    model_obj.eval()
    _MODELS[key] = (model_obj, tokenizer)
    return (model_obj, tokenizer)
end

function aiembed(
    schema::LocalManagedSchema,
    doc::AbstractString,
    postprocess::F=identity;
    verbose::Bool=true,
    model::AbstractString=DEFAULT_EMBEDDING,
    kwargs...
) where {F<:Function}
    m = load_embedding_model(model)
    docs_py = pylist([doc])

    time = @elapsed begin
        emb_py = m.encode(docs_py)
    end

    emb = vec(pyconvert(Matrix{Float32}, emb_py))
    msg = DataMessage(;
        content=postprocess(emb),
        status=200,
        cost=nothing,
        tokens=(0, 0),
        elapsed=time)
    verbose && @info "Local Embedding ($model): $(length(emb)) dims in $(round(time; digits=2))s"
    return msg
end

function aiembed(
    schema::LocalManagedSchema,
    docs::AbstractVector{<:AbstractString},
    postprocess::F=identity;
    verbose::Bool=true,
    model::AbstractString=DEFAULT_EMBEDDING,
    kwargs...
) where {F<:Function}
    m = load_embedding_model(model)
    docs_py = pylist(collect(docs))

    time = @elapsed begin
        emb_py = m.encode(docs_py)
    end

    emb_matrix = pyconvert(Matrix{Float32}, emb_py)
    n_docs = length(docs)
    dim = size(emb_matrix, 2)
    result = zeros(Float32, dim, n_docs)
    for i in 1:n_docs
        result[:, i] = postprocess(emb_matrix[i:i, :])
    end

    msg = DataMessage(;
        content=result,
        status=200,
        cost=nothing,
        tokens=(0, 0),
        elapsed=time)
    verbose && @info "Local Embedding ($model): $n_docs docs in $(round(time; digits=2))s"
    return msg
end

function aigenerate(
    schema::LocalManagedSchema,
    prompt::AbstractString;
    verbose::Bool=true,
    model::AbstractString=DEFAULT_GENERATION,
    max_new_tokens::Int=512,
    temperature::Float64=0.7,
    kwargs...
)
    gen_model, tokenizer = load_generation_model(model)
    torch = _py("torch")

    time = @elapsed begin
        inputs = tokenizer(prompt; return_tensors="pt")
        input_len = pylen(inputs["input_ids"][0])
        outputs = pywith(torch.no_grad()) do
            gen_model.generate(
                inputs["input_ids"];
                max_new_tokens=max_new_tokens,
                temperature=Float64(temperature),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        end
        tokens = outputs[0][input_len:end]
        text_py = tokenizer.decode(tokens; skip_special_tokens=true)
        text = pystr(text_py)
    end

    text = strip(text)
    msg = DataMessage(;
        content=text,
        status=200,
        cost=nothing,
        tokens=(0, 0),
        elapsed=time)
    verbose && @info "Local Generate ($model): $(length(text)) chars in $(round(time; digits=2))s"
    return msg
end

function aigenerate(
    schema::LocalManagedSchema,
    prompt::AbstractVector;
    verbose::Bool=true,
    model::AbstractString=DEFAULT_GENERATION,
    max_new_tokens::Int=512,
    temperature::Float64=0.7,
    kwargs...
)
    rendered = PromptingTools.render(schema, prompt)
    return aigenerate(schema, rendered;
        verbose, model, max_new_tokens, temperature)
end

end
