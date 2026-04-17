module Embedding

using ..Providers
using RAGTools

function build_index_rag(cfg, files; embedder::EmbeddingProvider, embedder_kwargs=())
    return RAGTools.build_index(cfg, files; embedder=embedder, embedder_kwargs=embedder_kwargs)
end

# For backward compatibility, but now deprecated
function build_index_rag(cfg, files; embedder_kwargs=())
    # Default to some embedder, but since RAGTools might expect different, perhaps error
    error("Please provide an embedder::EmbeddingProvider")
end

end