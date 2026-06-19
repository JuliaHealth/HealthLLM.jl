module Grounding

using PromptingTools
using PromptingTools: SystemMessage, UserMessage

const DEFAULT_GROUNDING_DIR = joinpath(@__DIR__, "..", "docs", "grounding")

const FUNSQL_SYSTEM_PROMPT = """You are a world-class Julia and SQL expert specializing in FunSQL.jl.
Translate natural language analytical questions into valid, executable FunSQL.jl queries against an OMOP CDM database.

## FunSQL Rules
- Use `From(:tablename)` to reference tables. Chain operations with `|>`.
- Use Julia symbols for columns: `:column_name`. Prefixed: `:table => :column`.
- SQL functions: `FunSQL.Fun("op", args...)` — e.g. `FunSQL.Fun("=", :a, :b)`, `FunSQL.Fun("YEAR", :date)`, `FunSQL.Fun("DATEDIFF", "d", :start, :end)`
- Aggregations: `Agg.count()`, `Agg.sum(:col)`, `Agg.avg(:col)`, `Agg.min(:col)`, `Agg.max(:col)` — each ends with `|> As(:alias)`
- Named subqueries/CTEs: pipeline `|> As(:name)`, then `Define(:name => subquery)`
- Joins: `Join(From(:table), on = FunSQL.Fun("=", :left_col, :right_col))`
- CASE: `FunSQL.Fun("CASE", cond1, res1, cond2, res2, ..., default)`
- LIKE: `FunSQL.Fun("LIKE", :col, "pattern")`. IN subquery: `FunSQL.Fun("IN", :col, From(...) |> Select(:col))`
- ORDER BY DESC: `Order(FunSQL.Fun("DESC", :col))`. Limit: `Limit(n)`. Offset: `Offset(n)`. Distinct: `Distinct()`.
- Literal tables: `From([(v1, v2), (v3, v4)])`. Union: `... |> Union(From(...))`.
- Window: `FunSQL.Fun("DENSE_RANK", FunSQL.Fun("OVER", FunSQL.Fun("ORDER", :col)))`.

## OMOP Tables
- `person` (person_id, gender_concept_id, year_of_birth, race_concept_id, care_site_id)
- `condition_era` (condition_era_id, person_id, condition_concept_id, condition_era_start_date, condition_era_end_date)
- `condition_occurrence` (condition_occurrence_id, person_id, condition_concept_id, condition_start_date, condition_end_date)
- `drug_era` (drug_era_id, person_id, drug_concept_id, drug_era_start_date, drug_era_end_date)
- `drug_exposure` (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date, drug_exposure_end_date)
- `procedure_occurrence` (procedure_occurrence_id, person_id, procedure_concept_id, procedure_date)
- `observation` (observation_id, person_id, observation_concept_id, observation_date, value_as_number)
- `measurement` (measurement_id, person_id, measurement_concept_id, measurement_date, value_as_number)
- `death` (person_id, death_date, death_type_concept_id, cause_concept_id)
- `visit_occurrence` (visit_occurrence_id, person_id, visit_concept_id, visit_start_date, visit_end_date)
- `care_site` (care_site_id, care_site_name, place_of_service_concept_id)
- `provider` (provider_id, provider_name, specialty_concept_id, care_site_id)
- `concept` (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, concept_code)
- `concept_ancestor` (ancestor_concept_id, descendant_concept_id, min_levels_of_separation, max_levels_of_separation)
- `concept_relationship` (concept_id_1, concept_id_2, relationship_id)
- `location` (location_id, city, state, zip)

Respond ONLY with the FunSQL.jl code. No explanations, no markdown. The code must be a valid Julia expression."""

function grounding_dir()
    return isdir(DEFAULT_GROUNDING_DIR) ? DEFAULT_GROUNDING_DIR : nothing
end

function grounding_files()
    dir = grounding_dir()
    dir === nothing && return String[]
    filter(endswith(".md"), readdir(dir, join=true))
end

function register_funsql_template!(; name::Symbol=:FunSQLQueryGeneration)
    if haskey(PromptingTools.TEMPLATE_STORE, name)
        return name
    end
    system_msg = FUNSQL_SYSTEM_PROMPT * "\n\n**Context Information:**\n---\n{{context}}\n---"
    template = [
        SystemMessage(system_msg, [:context], :systemmessage),
        UserMessage("# Question\n\n{{question}}\n\n# Answer (FunSQL code only)", [:question], nothing, :usermessage),
    ]
    PromptingTools.TEMPLATE_STORE[name] = template
    name
end

function register_funsql_template_no_context!(; name::Symbol=:FunSQLQueryDirect)
    if haskey(PromptingTools.TEMPLATE_STORE, name)
        return name
    end
    template = [
        SystemMessage(FUNSQL_SYSTEM_PROMPT, Symbol[], :systemmessage),
        UserMessage("# Question\n\n{{question}}\n\n# Answer (FunSQL code only)", [:question], nothing, :usermessage),
    ]
    PromptingTools.TEMPLATE_STORE[name] = template
    name
end

end
