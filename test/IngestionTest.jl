using HealthLLM
using HealthLLM.Ingestion: SourceDocument, SearchResult,
    AbstractSearchProvider, DuckDuckGoProvider,
    default_search_provider, web_search,
    CURATED_SOURCES, html_to_text, _title_from,
    Chunk, AbstractChunkStrategy, RecursiveChunk, HeaderChunk, RecordChunk, FixedSizeChunk,
    chunk, chunk_document, chunk_provenance, default_strategy, load_funsql_examples

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

    @testset "chunking strategies" begin
        @testset "RecursiveChunk" begin
            s = RecursiveChunk(; max_length=20)
            chunks = chunk(s, "First paragraph here.\n\nSecond paragraph here.")
            @test length(chunks) >= 2
            @test all(c -> c isa Chunk, chunks)
            @test all(c -> !isempty(c.text) && c.text == strip(c.text), chunks)
            # short text stays a single chunk
            @test only(chunk(RecursiveChunk(), "tiny")).text == "tiny"
            # empty in, empty out
            @test isempty(chunk(RecursiveChunk(), "   "))
            # parent heading is carried onto each chunk
            withhdr = chunk(RecursiveChunk(; max_length=15), "## person\nfield one two three four five")
            @test all(c -> c.metadata[:heading] == "person", withhdr)
        end

        @testset "RecursiveChunk preserves code blocks" begin
            code = "```julia\n" * repeat("From(:person) |> ", 40) * "Select(:x)\n```"
            text = "Intro paragraph.\n\n" * code * "\n\nOutro paragraph."
            chunks = chunk(RecursiveChunk(; max_length=80), text)
            # the FunSQL snippet is never split mid-expression: it lands whole in one chunk
            @test any(c -> occursin(code, c.text), chunks)
        end

        @testset "HeaderChunk" begin
            doc = "# Title\nintro\n\n## person\nperson_id, gender_concept_id\n\n## visit_occurrence\nvisit_occurrence_id, person_id"
            chunks = chunk(HeaderChunk(), doc)
            # one chunk per heading section (title + two tables)
            @test length(chunks) == 3
            person = only(filter(c -> c.metadata[:heading] == "person", chunks))
            @test occursin("person_id, gender_concept_id", person.text)
            # a table's columns are never split across chunks
            visit = only(filter(c -> get(c.metadata, :heading, "") == "visit_occurrence", chunks))
            @test occursin("visit_occurrence_id", visit.text) && occursin("person_id", visit.text)
            # min_level skips the title heading as a boundary; the two tables stay whole
            top = chunk(HeaderChunk(; min_level=2), doc)
            @test count(c -> get(c.metadata, :heading, "") in ("person", "visit_occurrence"), top) == 2
            # oversized section falls back to code-preserving recursive splitting
            big = "## t\n" * repeat("word ", 2000)
            bigchunks = chunk(HeaderChunk(; max_length=200), big)
            @test length(bigchunks) > 1
            @test all(c -> c.metadata[:heading] == "t", bigchunks)
            # a heading inside a code fence is not treated as a boundary
            fenced = "## real\nbody\n```\n## not-a-heading\ncode\n```"
            @test length(chunk(HeaderChunk(), fenced)) == 1
        end

        @testset "RecordChunk" begin
            jsonl = """
            {"query":"Count care sites","group":"care_site","description":"count","sql_query":"SELECT 1","response":"From(:care_site)"}
            {"query":"Patients per site","sql_query":"SELECT 2","response":"From(:person)"}
            """
            chunks = chunk(RecordChunk(), jsonl)
            # exactly one chunk per record, each self-contained
            @test length(chunks) == 2
            @test occursin("Query: Count care sites", chunks[1].text)
            @test occursin("FunSQL: From(:care_site)", chunks[1].text)
            # NL query leads the chunk so embeddings weight the question side
            @test startswith(chunks[1].text, "Query:")
            # parent-child metadata: the OMOP group/table and query are retained
            @test chunks[1].metadata[:group] == "care_site"
            @test chunks[1].metadata[:heading] == "Count care sites"
            # missing field (no description on record 2) is skipped, not blanked
            @test !occursin("Description:", chunks[2].text)
            # unparseable lines are dropped
            @test length(chunk(RecordChunk(), "not json\n{\"query\":\"q\"}")) == 1
        end

        @testset "FixedSizeChunk" begin
            text = repeat("x", 1000)
            chunks = chunk(FixedSizeChunk(; size=400, overlap=50), text)
            @test length(chunks) == 3
            @test length(chunks[1].text) == 400
            @test isempty(chunk(FixedSizeChunk(), ""))
            @test_throws AssertionError chunk(FixedSizeChunk(; size=10, overlap=10), "abc")
        end

        @testset "default_strategy dispatch" begin
            jsonl = SourceDocument("FunSQL-examples", "f.jsonl", "f", "{\"query\":\"q\",\"response\":\"r\"}")
            @test default_strategy(jsonl) isa RecordChunk
            omop = SourceDocument("OMOP CDM", "u", "t", "# a\nx\n## person\ncols")
            @test default_strategy(omop) isa HeaderChunk
            prose = SourceDocument("web-search", "u", "t", "just some flat prose without headings")
            @test default_strategy(prose) isa RecursiveChunk
            # content sniff: JSONL content routes to RecordChunk even off-name
            sniff = SourceDocument("misc", "u", "t", "{\"query\":\"q\",\"response\":\"r\"}")
            @test default_strategy(sniff) isa RecordChunk
        end

        @testset "chunk_document + provenance + load_funsql_examples" begin
            omop = SourceDocument("OMOP CDM", "http://cdm", "CDM", "## person\nperson_id\n\n## visit\nvisit_id")
            docchunks = chunk_document(omop)
            @test length(docchunks) == 2
            # document provenance is enriched onto every chunk
            @test docchunks[1].metadata[:source] == "OMOP CDM"
            @test docchunks[1].metadata[:url] == "http://cdm"
            @test docchunks[1].metadata[:heading] == "person"
            # provenance string ties the chunk back to its source and parent table
            @test chunk_provenance(docchunks[1]) == "http://cdm › person"
            @test length(chunk_provenance(docchunks[1])) <= 512
            # explicit strategy override is honoured
            @test length(chunk_document(omop; strategy=FixedSizeChunk(; size=1_000))) == 1

            mktemp() do path, io
                write(io, "{\"query\":\"q1\",\"group\":\"g\",\"response\":\"r1\"}\n{\"query\":\"q2\",\"response\":\"r2\"}\n")
                close(io)
                docs = load_funsql_examples(path)
                @test length(docs) == 1
                @test docs[1].source == "FunSQL-examples"
                exs = chunk_document(docs[1])
                @test length(exs) == 2
                @test exs[1].metadata[:group] == "g"
            end
            @test_throws ArgumentError load_funsql_examples(joinpath(tempdir(), "no_such_funsql_file.jsonl"))
        end
    end
end
