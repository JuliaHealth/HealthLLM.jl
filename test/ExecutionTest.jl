using HealthLLM.Execution: extract_funsql, generate_funsql, FunSQLGeneration,
    sanity_check_funsql, FunSQLCheck
using FunSQL

# A tiny OMOP-flavoured catalog to render/validate against.
const _CATALOG = FunSQL.SQLCatalog(
    FunSQL.SQLTable(:person, columns=[:person_id, :gender_concept_id, :year_of_birth]),
    FunSQL.SQLTable(:condition_occurrence,
        columns=[:condition_occurrence_id, :person_id, :condition_concept_id, :condition_start_date]),
    dialect=:duckdb,
)

@testset "Execution" begin
    @testset "extract_funsql" begin
        reply = """
        Here is the query:

        ```julia
        From(:person) |> Group() |> Select(:n => Agg.count())
        ```

        It counts patients.
        """
        code = extract_funsql(reply)
        @test occursin("From(:person)", code)
        @test !occursin("```", code)
        @test !occursin("It counts patients", code)     # explanation stripped

        # untagged fence still works
        @test extract_funsql("```\nFrom(:person)\n```") == "From(:person)"
        # no fence -> whole text, stripped
        @test extract_funsql("   From(:person)   ") == "From(:person)"
        # prefers the julia-tagged block over an earlier sql block
        mixed = "```sql\nSELECT 1\n```\n```julia\nFrom(:person)\n```"
        @test extract_funsql(mixed) == "From(:person)"
    end

    @testset "generate_funsql with injected generator" begin
        # Fake model: echoes a fenced FunSQL block, ignores the real LLM.
        fake = (prompt; model, schema=nothing, kwargs...) ->
            "```julia\nFrom(:person) |> Select(Get.person_id)\n```"
        p = (; system="S", user="U", prompt="S\n\nU")
        gen = generate_funsql(p; generator=fake, model="test-model")
        @test gen isa FunSQLGeneration
        @test gen.funsql == "From(:person) |> Select(Get.person_id)"
        @test occursin("```", gen.answer)               # raw answer keeps the fence

        # a plain-string prompt is accepted too
        gen2 = generate_funsql("just a string"; generator=fake, model="m")
        @test occursin("From(:person)", gen2.funsql)
    end

    @testset "sanity_check_funsql: parse stage" begin
        bad = sanity_check_funsql("From(:person) |> Select(")   # unbalanced
        @test !bad.ok
        @test !bad.parsed
        @test bad.stage == :none
        @test bad.error !== nothing
    end

    @testset "sanity_check_funsql: render against schema" begin
        ok = sanity_check_funsql(
            "From(:person) |> Group() |> Select(:n => Agg.count())"; catalog=_CATALOG)
        @test ok.ok
        @test ok.parsed && ok.built && ok.rendered
        @test ok.stage == :rendered
        @test ok.sql !== nothing
        @test occursin(r"(?i)select", ok.sql)

        # a join across two real tables also resolves
        joined = sanity_check_funsql(
            """
            From(:condition_occurrence) |>
                Join(:p => From(:person), Get.person_id .== Get.p.person_id) |>
                Group(Get.p.gender_concept_id) |>
                Select(Get.gender_concept_id, :n => Agg.count())
            """; catalog=_CATALOG)
        @test joined.ok
    end

    @testset "sanity_check_funsql: hallucinated names fail against schema" begin
        # column that isn't in the catalog
        bad_col = sanity_check_funsql(
            "From(:person) |> Select(Get.made_up_column)"; catalog=_CATALOG)
        @test !bad_col.ok
        @test bad_col.built                              # it built, but...
        @test !bad_col.rendered                          # ...failed to resolve
        @test bad_col.stage == :built
        @test bad_col.error !== nothing

        # table that isn't in the catalog
        bad_tbl = sanity_check_funsql("From(:not_a_table)"; catalog=_CATALOG)
        @test !bad_tbl.ok
        @test !bad_tbl.rendered
    end

    @testset "sanity_check_funsql: structural render without catalog" begin
        # No catalog: renders structurally against the dialect.
        res = sanity_check_funsql(
            "From(FunSQL.SQLTable(:person, columns=[:person_id])) |> Select(Get.person_id)";
            dialect=:duckdb)
        @test res.ok
        @test occursin(r"(?i)select", res.sql)
    end

    @testset "sanity_check_funsql: executor stage" begin
        code = "From(:person) |> Select(Get.person_id)"
        # executor that accepts the SQL -> executed
        seen = Ref{String}("")
        good = sanity_check_funsql(code; catalog=_CATALOG,
            executor=sql -> (seen[] = sql))
        @test good.ok
        @test good.executed
        @test good.stage == :executed
        @test occursin(r"(?i)select", seen[])

        # executor that rejects the SQL -> stops at :rendered
        boom = sanity_check_funsql(code; catalog=_CATALOG,
            executor=_ -> error("connection refused"))
        @test !boom.ok
        @test boom.rendered && !boom.executed
        @test boom.stage == :rendered
        @test occursin("connection refused", boom.error)
    end

    @testset "sanity_check_funsql on a FunSQLGeneration" begin
        gen = FunSQLGeneration("From(:person) |> Select(Get.person_id)", "raw")
        @test sanity_check_funsql(gen; catalog=_CATALOG).ok
    end
end
