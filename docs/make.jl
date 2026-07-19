using HealthLLM
using Documenter

DocMeta.setdocmeta!(HealthLLM, :DocTestSetup, :(using HealthLLM); recursive=true)

makedocs(;
    modules=[HealthLLM],
    authors="ParamThakkar123 <paramthakkar864@gmail.com> and TheCedarPrince <jacobszelko@gmail.com>",
    sitename="HealthLLM.jl",
    format=Documenter.HTML(;
        canonical="https://ParamThakkar123.github.io/HealthLLM.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Document Ingestion" => "ingestion.md",
        "Building Embeddings" => "embeddings.md",
        "Querying the RAG System" => "querying.md",
    ],
)

deploydocs(;
    repo="github.com/ParamThakkar123/HealthLLM.jl",
    devbranch="master",
)
