using HealthLLM
using Documenter

DocMeta.setdocmeta!(HealthLLM, :DocTestSetup, :(using HealthLLM); recursive=true)

makedocs(;
    modules=[HealthLLM],
    authors="ParamThakkar123 <paramthakkar864@gmail.com> and TheCedarPrince <jacobszelko@gmail.com>",
    sitename="HealthLLM.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaHealth.github.io/HealthLLM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaHealth/HealthLLM.jl",
    devbranch="main",
)
