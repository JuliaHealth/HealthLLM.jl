using HealthLLM
using HealthLLM.Ingestion: GROUNDING_DIR, load_grounding_docs, chunk_document, HeaderChunk, default_strategy
using HealthLLM.Execution: sanity_check_funsql
using FunSQL

# The grounding docs are the model's only admissible source of FunSQL syntax and
# OMOP column names (see FUNSQL_SYSTEM_PROMPT), so a wrong snippet in them is
# worse than none: it teaches the model a construct that doesn't exist. These
# tests render every FunSQL example against a catalog built from the schema doc
# itself, which also keeps the examples and the schema consistent.

# `## table` headings followed by "- `column` TYPE" bullets -> FunSQL catalog.
function _schema_catalog(path)
    tables = FunSQL.SQLTable[]
    name = nothing
    cols = Symbol[]
    flush!() = (name === nothing || isempty(cols) || push!(tables, FunSQL.SQLTable(name, columns=copy(cols))))
    for line in eachline(path)
        if (m = match(r"^##\s+([a-z_]+)\s*$", line)) !== nothing
            flush!()
            name = Symbol(m[1])
            empty!(cols)
        elseif (m = match(r"^-\s+`([a-z_0-9]+)`", line)) !== nothing && name !== nothing
            push!(cols, Symbol(m[1]))
        end
    end
    flush!()
    return FunSQL.SQLCatalog(tables..., dialect=:duckdb)
end

_julia_blocks(text) = [String(strip(m[1])) for m in eachmatch(r"```julia\r?\n(.*?)```"s, text)]

@testset "Grounding docs" begin
    schema_path = joinpath(GROUNDING_DIR, "omop_cdm_schema.md")
    catalog = _schema_catalog(schema_path)

    @testset "schema doc parses into a catalog" begin
        for t in (:person, :condition_occurrence, :condition_era, :drug_exposure,
            :drug_era, :visit_occurrence, :concept, :concept_ancestor, :death)
            @test haskey(catalog, t)
        end
        @test :person_id in keys(catalog[:person].columns)
    end

    for file in ("funsql_reference.md", "sql_to_funsql_patterns.md")
        @testset "every FunSQL example in $file renders" begin
            blocks = _julia_blocks(read(joinpath(GROUNDING_DIR, file), String))
            @test length(blocks) >= 15
            for code in blocks
                check = sanity_check_funsql(code; catalog=catalog)
                check.ok || @info "Grounding example failed to render" file code check.error
                @test check.ok
            end
        end
    end

    @testset "docs avoid constructs FunSQL does not have" begin
        for file in ("funsql_reference.md", "sql_to_funsql_patterns.md")
            for code in _julia_blocks(read(joinpath(GROUNDING_DIR, file), String))
                @test !occursin(r"\b(Distinct|Offset|Union|Having)\(", code)
                @test !occursin(r"DATEDIFF|DATEADD", code)
            end
        end
    end

    @testset "load_grounding_docs" begin
        docs = load_grounding_docs()
        @test length(docs) == 3
        @test all(d -> startswith(d.source, "grounding:"), docs)
        @test all(d -> !isempty(d.title), docs)
        @test all(d -> default_strategy(d) isa HeaderChunk, docs)

        # One chunk per OMOP table: the person table's columns stay together.
        schema = only(filter(d -> d.source == "grounding:omop_cdm_schema", docs))
        chunks = chunk_document(schema)
        person = only(filter(c -> get(c.metadata, :heading, "") == "person", chunks))
        @test occursin("person_id", person.text) && occursin("year_of_birth", person.text)
        @test !occursin("condition_occurrence_id", person.text)

        @test_throws ArgumentError load_grounding_docs(joinpath(GROUNDING_DIR, "no-such-dir"))
    end
end
