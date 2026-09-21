module HealthLLM

using PromptingTools
using RAGTools

# Included in dependency order: each module may use the ones above it.
include("huggingface.jl")
include("utils.jl")
include("database.jl")
include("embeddings.jl")
include("storage.jl")
include("prompt.jl")
include("execution.jl")
include("query.jl")
include("ingestion.jl")

"""
    HealthLLM.PUBLIC_MODULES

The submodules whose exported names make up the public API of `HealthLLM`.
Everything a submodule exports is re-exported from the package, so each symbol is
declared in exactly one place — the `export` list of the module that defines it.
Adding a name to that list is all it takes to publish it; there is no second
mirrored list here to fall out of sync.
"""
const PUBLIC_MODULES = (HuggingFace, Utils, Database, Embeddings, Storage, Prompt,
    Execution, Query, Ingestion)

# Explicit `import` rather than `using`, because an explicit import takes
# precedence over names a `using` brings in: `Storage.retrieve` has to win over
# the `retrieve` that `using RAGTools` above also provides, or the name resolves
# to neither and disappears from the package's API.
for m in PUBLIC_MODULES, n in names(m)
    n === nameof(m) && continue
    Core.eval(@__MODULE__, Meta.parse("import .$(nameof(m)): $n"))
    Core.eval(@__MODULE__, Expr(:export, n))
end

# The two upstream packages callers routinely need alongside HealthLLM.
export PromptingTools, RAGTools

end
