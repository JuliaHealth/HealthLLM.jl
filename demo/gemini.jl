using PromptingTools
using Statistics
using Printf
using GoogleGenAI

const COMBINED_PATH = raw"E:\HealthLLM.jl\demo\JuliaHealthLLM_exp_raw_combined.txt"
const GENAI_MODEL = get(ENV, "GENAI_MODEL", "gemini-2.5-flash")

if !haskey(ENV, "SSL_CERT_FILE")
    cacert_path = joinpath(@__DIR__, "cacert.pem")
    if !isfile(cacert_path)
        try
            using Downloads
            Downloads.download("https://curl.se/ca/cacert.pem", cacert_path)
        catch e
            @warn "Failed to download CA bundle: $e"
        end
    end
    if isfile(cacert_path)
        ENV["SSL_CERT_FILE"] = cacert_path
    end
end

function load_chunks(path::AbstractString; chunk_chars::Int=3000)
    txt = read(path, String)
    parts = split(txt, r"=== FILE:")
    chunks = String[]
    for p in parts
        p = strip(p)
        isempty(p) && continue
        cs = collect(p)
        pos = 1
        while pos <= length(cs)
            endpos = min(pos + chunk_chars - 1, length(cs))
            push!(chunks, strip(join(cs[pos:endpos])))
            pos = endpos + 1
        end
    end
    return chunks
end

# --- simple TF L2-normalized vector ---
function tf_vector(s::AbstractString)
    toks = [lowercase(t) for t in split(s, r"\W+") if !isempty(t)]
    m = Dict{String, Float64}()
    for t in toks
        m[t] = get(m, t, 0.0) + 1.0
    end
    norm = sqrt(sum(v->v^2, values(m)))
    if norm > 0
        for k in keys(m)
            m[k] /= norm
        end
    end
    return m
end

function cosine(u::Dict{String,Float64}, v::Dict{String,Float64})
    s = 0.0
    for (k,uv) in u
        s += uv * get(v, k, 0.0)
    end
    return s
end

function build_index(path::AbstractString)
    chunks = load_chunks(path)
    vecs = [tf_vector(c) for c in chunks]
    return (chunks=chunks, vecs=vecs)
end

function retrieve(index; query::AbstractString, topk::Int=5)
    qv = tf_vector(query)
    scores = [cosine(qv, v) for v in index.vecs]
    idxs = sortperm(scores; rev=true)[1:min(topk, length(scores))]
    return [(i, index.chunks[i], scores[i]) for i in idxs if scores[i] > 0.0]
end

function rag_answer(index; query::AbstractString, topk::Int=5)
    hits = retrieve(index; query=query, topk=topk)
    context = join([h[2] for h in hits], "\n\n---\n\n")
    prompt = """
    Use the following retrieved context to answer the question concisely.
    For any given query write its FunSQL query. Give the funsql query as output.
    Give only the FunSQL query as output nothing else.

    CONTEXT:
    $context

    QUESTION:
    $query

    ANSWER:
    """
    resp = aigenerate(prompt; model=GENAI_MODEL, api_key="AIzaSyBsoIiL8I1UPMCPQSmyTRfGG2VGG4nfzCw")
    return (answer=resp, retrieved=hits)
end

function main()
    @printf("Loading index from %s ...\n", COMBINED_PATH)
    index = build_index(COMBINED_PATH)
    @printf("Loaded %d chunks\n", length(index.chunks))

    println("Enter a question (empty line to exit):")
    while true
        print("> ")
        q = readline()
        isempty(strip(q)) && break
        try
            out = rag_answer(index; query=q, topk=5)
            println("\n--- Retrieved snippets (score) ---")
            for (i,txt,score) in out.retrieved
                @printf("[%d] %.4f\n", i, score)
                println(first(txt, 500), (length(txt)>500 ? "..." : ""))
                println()
            end
            println("\n--- Generated answer ---")
            println(out.answer)
            println("\n")
        catch err
            @printf("ERROR: %s\n", err)
        end
    end
    println("bye")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end