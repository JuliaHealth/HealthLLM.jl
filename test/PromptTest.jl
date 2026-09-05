using HealthLLM.Prompt: FUNSQL_SYSTEM_PROMPT, PromptTemplate, DEFAULT_FUNSQL_TEMPLATE,
    format_context, build_prompt

@testset "Prompt" begin
    @testset "build_prompt basics" begin
        hits = [
            (; index=1, chunk="# person\nperson_id, gender_concept_id, year_of_birth", score=0.91),
            (; index=2, chunk="Query: count patients\nFunSQL: From(person) |> Group() |> Select(n = Agg.count())", score=0.83),
        ]
        q = "How many patients are there?"
        p = build_prompt(q, hits)

        @test p isa NamedTuple
        @test p.system == FUNSQL_SYSTEM_PROMPT
        @test occursin(q, p.user)                       # question injected verbatim
        @test occursin("person_id", p.user)             # schema chunk injected
        @test occursin("From(person)", p.user)          # example chunk injected
        @test occursin("[Source 1]", p.user)            # numbered
        @test occursin("[Source 2]", p.user)
        @test p.prompt == string(p.system, "\n\n", p.user)
    end

    @testset "input flavours: strings, hits, Chunks" begin
        # plain strings
        u1 = build_prompt("q", ["alpha", "beta"]).user
        @test occursin("alpha", u1) && occursin("beta", u1)

        # pgvector-style hit uses .distance for the score slot
        tmpl = PromptTemplate(include_scores=true)
        u2 = build_prompt("q", [(; id=7, chunk="gamma", distance=0.25)]; template=tmpl).user
        @test occursin("gamma", u2)
        @test occursin("0.25", u2)

        # Chunk-like object: .text + .metadata provenance
        chunklike = (; text="delta", metadata=Dict{Symbol,Any}(:url => "u", :heading => "person"))
        u3 = build_prompt("q", [chunklike]).user
        @test occursin("delta", u3)
        @test occursin("u › person", u3)                # provenance rendered
    end

    @testset "empty / whitespace handling" begin
        @test_throws ArgumentError build_prompt("   ", ["x"])

        # no context -> the note, not a blank section
        p = build_prompt("q")
        @test occursin(DEFAULT_FUNSQL_TEMPLATE.empty_context_note, p.user)

        # blank chunks are dropped
        @test format_context(DEFAULT_FUNSQL_TEMPLATE, ["", "   ", "real"]) |>
              c -> occursin("real", c) && occursin("[Source 1]", c) && !occursin("[Source 2]", c)
    end

    @testset "budgets: max_chunks and max_context_chars" begin
        many = ["chunk-$i body" for i in 1:20]
        tmpl = PromptTemplate(max_chunks=3)
        ctx = format_context(tmpl, many)
        @test occursin("[Source 3]", ctx)
        @test !occursin("[Source 4]", ctx)

        # char budget stops at a chunk boundary
        small = PromptTemplate(max_context_chars=40, max_chunks=100)
        ctx2 = format_context(small, ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd"])
        @test length(ctx2) <= 120                       # well under an unbudgeted join
        @test occursin("[Source 1]", ctx2)

        # a lone over-budget chunk is truncated, not dropped
        ctx3 = format_context(small, [repeat("z", 500)])
        @test occursin("…", ctx3)
        @test length(ctx3) < 500
    end

    @testset "template customisation" begin
        tmpl = PromptTemplate(system="SYS", answer_cue="", chunk_label="Doc")
        p = build_prompt("q", ["ctx"]; template=tmpl)
        @test p.system == "SYS"
        @test occursin("[Doc 1]", p.user)
        @test !occursin("# FunSQL query", p.user)       # answer cue omitted
    end
end
