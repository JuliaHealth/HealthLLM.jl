using HealthLLM
using HealthLLM.Ingestion: SourceDocument, SearchResult,
    AbstractSearchProvider, DuckDuckGoProvider,
    default_search_provider, web_search,
    CURATED_SOURCES, html_to_text, _title_from

# These tests exercise the offline/pure logic only. Live fetching and search
# depend on the network and provider credentials and are not run in CI.

@testset "Ingestion" begin
    @testset "html_to_text" begin
        @test html_to_text("<p>Hello <b>world</b></p>") == "Hello world"
        @test html_to_text("<div>a</div><div>b</div>") == "a\nb"
        # script/style content is dropped
        @test !occursin("alert", html_to_text("<script>alert(1)</script>text"))
        @test occursin("text", html_to_text("<script>alert(1)</script>text"))
        # entity decoding
        @test occursin("&", html_to_text("A &amp; B"))
        @test occursin("<", html_to_text("&lt;tag&gt;"))
        @test html_to_text("&#65;&#66;") == "AB"
        # whitespace collapse
        @test !occursin("\n\n\n", html_to_text("a<br><br><br><br>b"))
    end

    @testset "curated registry" begin
        @test haskey(CURATED_SOURCES, "FunSQL.jl")
        @test haskey(CURATED_SOURCES, "OMOP CDM")
        @test haskey(CURATED_SOURCES, "OHDSI")
        @test haskey(CURATED_SOURCES, "JuliaHealth")
        @test all(!isempty, values(CURATED_SOURCES))
        @test all(u -> startswith(u, "http"), Iterators.flatten(values(CURATED_SOURCES)))
    end

    @testset "_title_from" begin
        @test _title_from("# Big Title\n\nbody", "u") == "Big Title"
        @test _title_from("no heading here\nmore", "u") == "no heading here"
        @test _title_from("   ", "http://x") == "http://x"
    end

    @testset "search provider" begin
        @test DuckDuckGoProvider() isa AbstractSearchProvider
        @test default_search_provider() isa DuckDuckGoProvider
    end

    @testset "SourceDocument / SearchResult constructors" begin
        d = SourceDocument("s", "u", "t", "c")
        @test d.source == "s" && d.content == "c"
        r = SearchResult("t", "u", "c", 0.5)
        @test r.score == 0.5
    end
end
