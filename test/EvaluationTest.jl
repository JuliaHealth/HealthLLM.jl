using HealthLLM
using HealthLLM.Evaluation: BenchmarkTask, TIERS, classify_tier, load_tasks, save_tasks,
    split_tasks, examples_document, ResultSet, results_match, ZERO_SHOT_SYSTEM_PROMPT,
    zero_shot_prompt, TaskResult, OUTCOMES, run_task, run_benchmark, summarize,
    compare_modes, mcnemar_exact, write_results, markdown_report
using HealthLLM.Prompt: FUNSQL_SYSTEM_PROMPT
using HealthLLM.Ingestion: chunk_document, RecordChunk, default_strategy
using FunSQL
using JSON3

# Offline tests: the database is a lookup table from SQL to a canned result, and
# the model is a function that answers from a script. FunSQL itself is real, so
# the parse -> build -> render stages are exercised for real.

const _EVAL_CATALOG = FunSQL.SQLCatalog(
    FunSQL.SQLTable(:person, columns=[:person_id, :gender_concept_id, :year_of_birth]),
    dialect=:duckdb,
)

_task(id, q; tier=:simple, sql="SELECT 1", funsql="From(:person)") =
    BenchmarkTask(id, q, "person", tier, sql, funsql, "")

@testset "Evaluation" begin
    @testset "classify_tier" begin
        @test classify_tier("SELECT COUNT(*) FROM cdm.person") === :simple
        @test classify_tier("SELECT gender_concept_id, COUNT(*) FROM person GROUP BY 1") === :simple
        @test classify_tier("SELECT * FROM person p JOIN death d ON p.person_id = d.person_id") === :compositional
        @test classify_tier("WITH t AS (SELECT 1) SELECT * FROM t") === :compositional
        @test classify_tier("SELECT * FROM person WHERE person_id IN (SELECT person_id FROM death)") === :compositional
        @test classify_tier("SELECT row_number() OVER (ORDER BY x) FROM t") === :compositional
        # vocabulary tables win over syntax
        @test classify_tier("SELECT * FROM cdm.concept WHERE concept_code = '1'") === :schema_aware
        @test classify_tier("SELECT * FROM condition_era ce JOIN cdm.concept_ancestor ca ON 1=1") === :schema_aware
        # column names that merely contain "concept" don't count
        @test classify_tier("SELECT gender_concept_id FROM person") === :simple
    end

    @testset "load_tasks / save_tasks round trip" begin
        dir = mktempdir()
        src = joinpath(dir, "raw.jsonl")
        open(src, "w") do io
            println(io, JSON3.write((query="Count persons", group="person",
                sql_query="SELECT COUNT(*) FROM person", response="From(:person)")))
            println(io, "")                                        # blank line skipped
            println(io, JSON3.write((query="", sql_query="SELECT 1")))  # no question -> skipped
            println(io, JSON3.write((query="Concept lookup", group="concept", tier="schema_aware",
                sql_query="SELECT 1", response="", id="custom-id")))
        end
        tasks = load_tasks(src)
        @test length(tasks) == 2
        @test tasks[1].id == "task-1"
        @test tasks[1].tier === :simple
        @test tasks[2].id == "custom-id"
        @test tasks[2].tier === :schema_aware        # explicit tier respected

        @test length(load_tasks(src; tiers=(:schema_aware,))) == 1

        out = save_tasks(joinpath(dir, "saved.jsonl"), tasks)
        again = load_tasks(out)
        @test [(t.id, t.question, t.tier, t.gold_sql, t.gold_funsql) for t in again] ==
              [(t.id, t.question, t.tier, t.gold_sql, t.gold_funsql) for t in tasks]

        bad = joinpath(dir, "bad.jsonl")
        write(bad, JSON3.write((query="q", sql_query="s", tier="impossible")))
        @test_throws ArgumentError load_tasks(bad)
        dup = joinpath(dir, "dup.jsonl")
        write(dup, JSON3.write((id="a", query="q", sql_query="s")) * "\n" *
                   JSON3.write((id="a", query="r", sql_query="s")))
        @test_throws ArgumentError load_tasks(dup)
        @test_throws ArgumentError load_tasks(joinpath(dir, "missing.jsonl"))
    end

    @testset "split_tasks is stratified, disjoint and deterministic" begin
        tasks = vcat([_task("s$i", "q"; tier=:simple) for i in 1:10],
            [_task("c$i", "q"; tier=:compositional) for i in 1:6],
            [_task("v$i", "q"; tier=:schema_aware) for i in 1:4])
        sp = split_tasks(tasks; test_fraction=0.5)
        @test length(sp.test) + length(sp.examples) == length(tasks)
        @test isempty(intersect(Set(t.id for t in sp.test), Set(t.id for t in sp.examples)))
        for tier in TIERS
            n = count(t -> t.tier === tier, tasks)
            @test count(t -> t.tier === tier, sp.test) == round(Int, n / 2)
        end
        @test [t.id for t in split_tasks(tasks).test] == [t.id for t in sp.test]
        # input order doesn't matter, the seed does
        @test Set(t.id for t in split_tasks(reverse(tasks)).test) == Set(t.id for t in sp.test)
        @test Set(t.id for t in split_tasks(tasks; seed=1).test) != Set(t.id for t in sp.test)
        @test_throws ArgumentError split_tasks(tasks; test_fraction=0)
    end

    @testset "examples_document chunks one example per task" begin
        tasks = [_task("a", "Count persons"; funsql="From(:person) |> Group() |> Select(:n => Agg.count())"),
            _task("b", "No reference"; funsql=""),          # skipped: nothing to teach
            _task("c", "Persons born after 1990"; funsql="From(:person) |> Where(Get.year_of_birth .> 1990)")]
        doc = examples_document(tasks)
        @test default_strategy(doc) isa RecordChunk
        chunks = chunk_document(doc)
        @test length(chunks) == 2
        @test occursin("Count persons", chunks[1].text) && occursin("Agg.count()", chunks[1].text)
        @test !any(c -> occursin("No reference", c.text), chunks)
    end

    @testset "results_match" begin
        gold = ResultSet(["gender", "n"], [[8507, 10], [8532, 12]])
        @test results_match(gold, gold)
        # row order ignored by default, respected when asked
        swapped_rows = ResultSet(["gender", "n"], [[8532, 12], [8507, 10]])
        @test results_match(gold, swapped_rows)
        @test !results_match(gold, swapped_rows; ordered=true)
        # different aliases, same content
        @test results_match(gold, ResultSet(["gender_concept_id", "count"], [[8507, 10], [8532, 12]]))
        # columns in a different order
        @test results_match(gold, ResultSet(["n", "gender"], [[10, 8507], [12, 8532]]))
        # numeric types and float noise
        @test results_match(gold, ResultSet(["gender", "n"], [[8507.0, 10.0000001], [8532, 12]]))
        # wrong value, extra row, extra column
        @test !results_match(gold, ResultSet(["gender", "n"], [[8507, 11], [8532, 12]]))
        @test !results_match(gold, ResultSet(["gender", "n"], [[8507, 10], [8532, 12], [0, 1]]))
        @test !results_match(gold, ResultSet(["gender", "n", "x"], [[8507, 10, 1], [8532, 12, 1]]))
        # missing and nothing are both NULL
        @test results_match(ResultSet(["a"], [[missing]]), ResultSet(["a"], [[nothing]]))
        @test !results_match(ResultSet(["a"], [[missing]]), ResultSet(["a"], [[0]]))
        # a value compares equal to its string form (e.g. a Date vs its ISO string)
        @test results_match(ResultSet(["d"], [["2020-01-01"]]), ResultSet(["d"], [[Base.Text("2020-01-01")]]))
        # empty results with the same shape match
        @test results_match(ResultSet(["a"], Vector{Any}[]), ResultSet(["b"], Vector{Any}[]))
    end

    @testset "zero_shot_prompt" begin
        p = zero_shot_prompt("How many persons?")
        @test p.system == ZERO_SHOT_SYSTEM_PROMPT
        @test occursin("How many persons?", p.user)
        @test p.prompt == string(p.system, "\n\n", p.user)
        # the baseline must not demand retrieved context it never gets
        @test !occursin("ONLY the retrieved context", ZERO_SHOT_SYSTEM_PROMPT)
        @test occursin("ONLY the retrieved context", FUNSQL_SYSTEM_PROMPT)
        @test_throws ArgumentError zero_shot_prompt("  ")
    end

    # --- run_task with a fake model and a fake database --------------------------------
    gold_sql = "SELECT COUNT(*) AS n FROM person"
    count_rs = ResultSet(["n"], [[42]])
    good_code = "From(:person) |> Group() |> Select(:n => Agg.count())"
    good_sql = string(FunSQL.render(_EVAL_CATALOG,
        FunSQL.From(:person) |> FunSQL.Group() |> FunSQL.Select(:n => FunSQL.Agg.count())))

    # Unknown SQL raises, like a database would.
    db = Dict(gold_sql => count_rs, good_sql => count_rs)
    executor = sql -> haskey(db, sql) ? db[sql] : error("no such table")
    reply(code) = "```julia\n$code\n```\nCounts persons."
    fixed(code) = (prompt; model, schema=nothing, kwargs...) -> reply(code)
    task = _task("t1", "How many persons are there?"; sql=gold_sql)
    common = (; model="fake", executor=executor, catalog=_EVAL_CATALOG)

    @testset "run_task outcomes" begin
        r = run_task(task, :zero_shot; generator=fixed(good_code), common...)
        @test r isa TaskResult
        @test r.outcome === :correct
        @test r.mode === :zero_shot
        @test r.funsql == good_code
        @test r.sql == good_sql
        @test isempty(r.context)

        @test run_task(task, :zero_shot; generator=fixed("From(:person) |> Select("), common...).outcome === :parse_error
        @test run_task(task, :zero_shot; generator=fixed("1 + 1"), common...).outcome === :build_error
        @test run_task(task, :zero_shot; generator=fixed("From(:person) |> Select(Get.made_up)"), common...).outcome === :render_error
        # renders, but the database has never heard of the query
        @test run_task(task, :zero_shot; generator=fixed("From(:person) |> Select(Get.person_id)"), common...).outcome === :execute_error
        # runs, wrong answer
        wrong = merge(common, (; executor=sql -> sql == gold_sql ? count_rs : ResultSet(["n"], [[7]])))
        r = run_task(task, :zero_shot; generator=fixed(good_code), wrong...)
        @test r.outcome === :mismatch
        @test r.error !== nothing
        # the model call fails
        boom = (prompt; kwargs...) -> error("model offline")
        r = run_task(task, :zero_shot; generator=boom, common...)
        @test r.outcome === :generation_error
        @test occursin("model offline", r.error)
        # the gold SQL fails: excluded, the model is never called
        called = Ref(false)
        spy = (prompt; kwargs...) -> (called[] = true; reply(good_code))
        r = run_task(_task("t2", "q"; sql="SELECT broken"), :zero_shot; generator=spy, common...)
        @test r.outcome === :gold_error
        @test !called[]

        @test_throws ArgumentError run_task(task, :fancy; generator=fixed(good_code), common...)
        @test_throws ArgumentError run_task(task, :rag; generator=fixed(good_code), common...)  # no retriever
    end

    @testset "run_task: RAG sends retrieved context to the model" begin
        seen = Ref{Any}(nothing)
        capture = (prompt; kwargs...) -> (seen[] = prompt; reply(good_code))
        hits = [(; text="## person\n- `person_id`", metadata=Dict{Symbol,Any}(:source => "grounding:omop_cdm_schema", :heading => "person"), score=0.9),
            (; chunk="Query: count persons\nFunSQL: From(:person)", score=0.7)]
        r = run_task(task, :rag; generator=capture, retriever=q -> hits, common...)
        @test r.outcome === :correct
        @test seen[].system == FUNSQL_SYSTEM_PROMPT
        @test occursin("person_id", seen[].user)
        @test r.context == ["grounding:omop_cdm_schema › person", "Query: count persons FunSQL: From(:person)"]
        @test r.scores == [0.9, 0.7]

        # zero-shot sends no context and the baseline system prompt
        run_task(task, :zero_shot; generator=capture, common...)
        @test seen[].system == ZERO_SHOT_SYSTEM_PROMPT
        @test !occursin("person_id", seen[].user)
    end

    # Model that is right only when the prompt carries retrieved context.
    rag_helps = (prompt; kwargs...) ->
        occursin("Retrieved context", prompt.user) ? reply(good_code) : reply("From(:person) |> Select(Get.made_up)")
    tasks = [_task("t$i", "How many persons? ($i)"; sql=gold_sql) for i in 1:8]
    push!(tasks, _task("broken", "q"; sql="SELECT broken"))
    results = run_benchmark(tasks; generator=rag_helps, retriever=q -> ["## person\n- `person_id`"],
        progress=false, common...)

    @testset "run_benchmark" begin
        @test length(results) == 2 * length(tasks)
        @test [r.mode for r in results[1:2]] == [:zero_shot, :rag]
        @test count(r -> r.mode === :rag && r.outcome === :correct, results) == 8
        @test count(r -> r.mode === :zero_shot && r.outcome === :render_error, results) == 8
        @test count(r -> r.outcome === :gold_error, results) == 2
        # zero-shot only
        zs = run_benchmark(tasks[1:2]; modes=(:zero_shot,), generator=rag_helps, progress=false, common...)
        @test all(r -> r.mode === :zero_shot, zs)
    end

    @testset "summarize" begin
        s = summarize(results)
        @test s[:rag].overall.tasks == 8                     # gold failure excluded
        @test s[:rag].excluded == 1
        @test s[:rag].overall.execution_accuracy == 1.0
        @test s[:zero_shot].overall.execution_accuracy == 0.0
        @test s[:zero_shot].overall.build_success == 1.0     # it built...
        @test s[:zero_shot].overall.render_success == 0.0    # ...but named a column that doesn't exist
        @test s[:zero_shot].overall.outcomes["render_error"] == 8
        @test s[:rag].by_tier["simple"].tasks == 8
        @test s[:rag].by_group["person"].correct == 8
    end

    @testset "compare_modes and mcnemar_exact" begin
        c = compare_modes(results)
        @test c.paired == 8
        @test c.treatment_only == 8 && c.baseline_only == 0
        @test c.both_correct == 0 && c.neither == 0
        @test c.delta == 1.0
        @test c.p_value == round(2 / 2^8; digits=4)
        @test length(c.treatment_only_ids) == 8

        @test mcnemar_exact(0, 0) == 1.0
        @test mcnemar_exact(5, 5) == 1.0
        @test mcnemar_exact(0, 8) ≈ 2 / 2^8
        @test mcnemar_exact(1, 9) ≈ 2 * (1 + 10) / 2^10
        @test mcnemar_exact(3, 4) == mcnemar_exact(4, 3)
    end

    @testset "write_results and markdown_report" begin
        path = write_results(joinpath(mktempdir(), "run.json"), results; meta=(; model="fake"))
        saved = JSON3.read(read(path, String))
        @test saved.meta.model == "fake"
        @test length(saved.results) == length(results)
        @test saved.comparison.paired == 8
        @test haskey(saved.summary, :rag)

        md = markdown_report(results; meta=(; model="fake"))
        @test occursin("Zero-shot vs RAG", md)
        @test occursin("| Execution accuracy | 0.0% | 100.0% |", md)
        @test occursin("render_error", md)
        @test occursin("1 task(s) excluded", md)
        @test occursin("Solved only with RAG", md)

        # a single-mode report still renders
        zs = filter(r -> r.mode === :zero_shot, results)
        @test !occursin("Zero-shot vs RAG", markdown_report(zs))
    end
end
