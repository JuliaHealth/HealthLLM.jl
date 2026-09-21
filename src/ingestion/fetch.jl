const _DEFAULT_HEADERS = ["User-Agent" => "HealthLLM.jl-ingestion/0.1"]

"""
    html_to_text(html) -> String

Strip HTML markup to readable plain text without an external parser.

Removes `<script>`/`<style>`/`<head>` blocks and HTML comments, converts a few
block-level tags to newlines, drops all remaining tags, decodes common HTML
entities, and collapses excess whitespace.

# Example

```julia
html_to_text("<p>Hello <b>world</b></p>")  # -> "Hello world"
```
"""
function html_to_text(html::AbstractString)
    s = String(html)
    s = replace(s, r"(?is)<script\b.*?</script>" => " ")
    s = replace(s, r"(?is)<style\b.*?</style>" => " ")
    s = replace(s, r"(?is)<head\b.*?</head>" => " ")
    s = replace(s, r"(?s)<!--.*?-->" => " ")
    s = replace(s, r"(?i)<br\s*/?>" => "\n")
    s = replace(s, r"(?i)</(p|div|li|tr|h[1-6]|section|article)>" => "\n")
    s = replace(s, r"<[^>]+>" => " ")
    for (pat, rep) in (
        "&nbsp;" => " ", "&amp;" => "&", "&lt;" => "<", "&gt;" => ">",
        "&quot;" => "\"", "&#39;" => "'", "&apos;" => "'", "&mdash;" => "—",
        "&ndash;" => "–", "&hellip;" => "…",
    )
        s = replace(s, pat => rep)
    end
    s = replace(s, r"&#(\d+);" => m -> string(Char(parse(Int, m[3:end-1]))))
    s = replace(s, r"[ \t]+" => " ")
    s = replace(s, r"\n[ \t]+" => "\n")
    s = replace(s, r"\n{3,}" => "\n\n")
    return strip(s)
end

_looks_like_markup(url, ctype) =
    occursin("html", lowercase(ctype)) ||
    (isempty(ctype) && !occursin(r"\.(md|markdown|txt|rst|csv|json)$"i, url))

"""
    fetch_url(url; timeout=30, max_bytes=5_000_000) -> String

Fetch `url` over HTTP(S) and return cleaned text.

Content served as HTML is run through [`html_to_text`](@ref); Markdown/plain
text is returned as-is. Responses larger than `max_bytes` are truncated. Throws
on network errors or non-2xx status.

# Keywords
- `timeout=30`: Per-request timeout in seconds.
- `max_bytes=5_000_000`: Byte cap on the response body before cleaning.
"""
function fetch_url(url::AbstractString; timeout::Real=30, max_bytes::Integer=5_000_000)
    resp = HTTP.get(String(url); headers=_DEFAULT_HEADERS, readtimeout=timeout,
        redirect=true, retry=false, status_exception=true)
    ctype = HTTP.header(resp, "Content-Type", "")
    body = String(resp.body)
    length(body) > max_bytes && (body = body[1:max_bytes])
    return _looks_like_markup(url, ctype) ? html_to_text(body) : strip(body)
end

function _title_from(content::AbstractString, url::AbstractString)
    m = match(r"(?m)^#{1,6}\s+(.+)$", content)
    m !== nothing && return strip(m[1])
    for line in eachline(IOBuffer(content))
        !isempty(strip(line)) && return first(strip(line), 120)
    end
    return String(url)
end

"""
    fetch_curated(names=keys(CURATED_SOURCES); timeout=30, min_length=200) -> Vector{SourceDocument}

Fetch the curated documentation sources named in `names` from [`CURATED_SOURCES`](@ref).

For each source, its candidate URLs are tried in order and every one that
fetches successfully and yields at least `min_length` characters is kept. URLs
that error are skipped with a warning. Returns a flat vector of
[`SourceDocument`](@ref).

# Example

```julia
docs = fetch_curated(["FunSQL.jl", "OMOP CDM"])
```
"""
function fetch_curated(names=keys(CURATED_SOURCES); timeout::Real=30, min_length::Integer=200)
    docs = SourceDocument[]
    for name in names
        haskey(CURATED_SOURCES, name) ||
            (@warn "Unknown curated source, skipping: $name"; continue)
        for url in CURATED_SOURCES[name]
            try
                content = fetch_url(url; timeout=timeout)
                if length(content) >= min_length
                    push!(docs, SourceDocument(name, url, _title_from(content, url), content))
                else
                    @debug "Skipping short/empty content ($(length(content)) chars) from $name: $url"
                end
            catch err
                @warn "Failed to fetch curated URL from $name, skipping: $url ($err)"
            end
        end
    end
    return docs
end
