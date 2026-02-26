module Embedding

using RAGTools

"""
    build_index_rag(files; embedder_kwargs=())

Build a RAG index using RAGTools.
"""
function build_index_rag(files; embedder_kwargs=())
    return RAGTools.build_index(files; embedder_kwargs=embedder_kwargs)
end

end
