module Embedding

using RAGTools

"""
    build_index_rag(files; embedder_kwargs=())

Build a RAG index from files using RAGTools for retrieval-augmented generation.

# Arguments
- `files`: Vector of file paths to build the index from.

# Keywords
- `embedder_kwargs=()`: Additional keyword arguments passed to the embedder.

# Returns
A RAG index object for querying.

# Example

```julia
files = ["document1.md", "document2.md"]
index = build_index_rag(files)
```
"""
function build_index_rag(files; embedder_kwargs=())
    return RAGTools.build_index(files; embedder_kwargs=embedder_kwargs)
end
end