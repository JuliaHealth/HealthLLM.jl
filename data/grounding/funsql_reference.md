# FunSQL.jl Reference

FunSQL.jl builds SQL queries by composing Julia values with the pipe operator
`|>`. A query is a single Julia expression that starts from a table and passes
through a chain of operations. FunSQL renders it to SQL for a target dialect
(DuckDB, PostgreSQL, ...). Every example below is a complete, valid query over
the OMOP CDM tables.

## Core rules

- Start from a table with `From(:table_name)`.
- Refer to a column with `Get.column_name`. Inside `Select`, `Group`, and
  `Order`, a bare symbol such as `:person_id` also works.
- Name an output column with `:name => expression`, or with `expression |> As(:name)`.
- Compare with broadcasting operators: `.==`, `.!=`, `.<`, `.>`, `.<=`, `.>=`.
  Combine conditions with `.&` (AND) and `.|` (OR), or with `Fun.and` / `Fun.or`.
- Call any SQL function with `Fun.function_name(args...)`, e.g. `Fun.lower(Get.concept_name)`.
- Aggregates live under `Agg`: `Agg.count()`, `Agg.count_distinct(x)`,
  `Agg.sum(x)`, `Agg.avg(x)`, `Agg.min(x)`, `Agg.max(x)`. They are only valid
  after `Group(...)` or `Partition(...)`.
- FunSQL has no `Distinct`, `Offset`, `Union`, or `Having` nodes. Use `Group`
  for DISTINCT, `Limit(start:stop)` for OFFSET, `Append` for UNION ALL, and
  `Where` after `Group` for HAVING.

## From and Select

```julia
From(:person) |>
Select(Get.person_id, Get.year_of_birth)
```

## Where

```julia
From(:person) |>
Where(Get.year_of_birth .> 1990) |>
Select(Get.person_id, Get.gender_concept_id)
```

Several conditions:

```julia
From(:person) |>
Where((Get.year_of_birth .>= 1950) .& (Get.gender_concept_id .== 8507)) |>
Select(Get.person_id)
```

## Group and aggregate

`Group` with keys produces one row per key. `Group()` with no keys aggregates the
whole table into one row.

```julia
From(:person) |>
Group(Get.gender_concept_id) |>
Select(Get.gender_concept_id, :n_persons => Agg.count())
```

```julia
From(:condition_era) |>
Group() |>
Select(:n_patients => Agg.count_distinct(Get.person_id))
```

## Filter on an aggregate (HAVING)

A `Where` placed after `Group` filters the groups:

```julia
From(:condition_era) |>
Group(Get.person_id) |>
Where(Agg.count() .> 5) |>
Select(Get.person_id, :n_eras => Agg.count())
```

## Distinct values

`Group` on the columns, with no aggregate, gives distinct rows:

```julia
From(:drug_exposure) |>
Group(Get.drug_concept_id) |>
Select(Get.drug_concept_id)
```

## Order and Limit

```julia
From(:person) |>
Order(Get.year_of_birth |> Desc()) |>
Limit(10) |>
Select(Get.person_id, Get.year_of_birth)
```

Skip rows with a range: `Limit(11:20)` returns rows 11 to 20.

## Join

Name the joined table with `:alias => From(...)`, then refer to its columns
through the alias as `Get.alias.column`. Columns of the left table stay
available as `Get.column`.

```julia
From(:condition_occurrence) |>
Join(:p => From(:person), Get.person_id .== Get.p.person_id) |>
Select(Get.condition_occurrence_id, Get.p.year_of_birth)
```

Look up a concept name. After `Group`, the grouping key is available by its
column name (`Get.concept_name`), not through the join alias:

```julia
From(:person) |>
Join(:c => From(:concept), Get.gender_concept_id .== Get.c.concept_id) |>
Group(Get.c.concept_name) |>
Select(:gender => Get.concept_name, :n_persons => Agg.count())
```

`LeftJoin` keeps unmatched rows from the left table:

```julia
From(:person) |>
LeftJoin(:d => From(:death), Get.person_id .== Get.d.person_id) |>
Select(Get.person_id, Get.d.death_date)
```

Always qualify the right-hand column with the alias. Writing
`Fun("=", :person_id, :person_id)` resolves both sides to the same column, which
silently produces the wrong join.

## Computed columns with Define

`Define` adds a named column that later steps can use:

```julia
From(:person) |>
Define(:age => 2024 .- Get.year_of_birth) |>
Where(Get.age .>= 65) |>
Select(Get.person_id, Get.age)
```

## Subqueries and IN

```julia
From(:condition_era) |>
Where(Fun.in(Get.condition_concept_id,
    From(:concept_ancestor) |>
    Where(Get.ancestor_concept_id .== 201826) |>
    Select(Get.descendant_concept_id))) |>
Group() |>
Select(:n_patients => Agg.count_distinct(Get.person_id))
```

`Fun.in` also takes a list of values: `Fun.in(Get.gender_concept_id, 8507, 8532)`.

## Common table expressions (WITH)

`With(:name => query)` defines a reusable subquery that `From(:name)` reads:

```julia
From(:diabetic) |>
Group() |>
Select(:n_patients => Agg.count_distinct(Get.person_id)) |>
With(:diabetic =>
    From(:condition_era) |>
    Join(:ca => From(:concept_ancestor),
        Get.condition_concept_id .== Get.ca.descendant_concept_id) |>
    Where(Get.ca.ancestor_concept_id .== 201826) |>
    Select(Get.person_id))
```

## Aggregate of an aggregate

Group twice. The second step reads the columns the first `Select` produced:

```julia
From(:condition_era) |>
Group(Get.person_id) |>
Select(Get.person_id, :n_eras => Agg.count()) |>
Group() |>
Select(:avg_eras_per_person => Agg.avg(Get.n_eras))
```

## Window functions

`Partition` makes aggregates and ranking functions act as window functions:

```julia
From(:drug_exposure) |>
Partition(Get.person_id, order_by = [Get.drug_exposure_start_date]) |>
Select(Get.person_id, Get.drug_concept_id, :exposure_number => Agg.row_number())
```

## Set operations

`Append` is UNION ALL. Both sides must select the same columns:

```julia
From(:condition_occurrence) |>
Select(Get.person_id) |>
Append(From(:drug_exposure) |> Select(Get.person_id))
```

## Conditional expressions

```julia
From(:person) |>
Select(Get.person_id,
    :birth_cohort => Fun.case(Get.year_of_birth .< 1950, "before 1950",
                              Get.year_of_birth .< 1980, "1950-1979",
                              "1980 or later"))
```

## NULL checks and ranges

```julia
From(:visit_occurrence) |>
Where(Fun.is_not_null(Get.care_site_id) .& Fun.between(Get.visit_start_date, Fun.cast("2020-01-01", "DATE"), Fun.cast("2020-12-31", "DATE"))) |>
Select(Get.visit_occurrence_id, Get.care_site_id)
```

## Text matching

```julia
From(:concept) |>
Where(Fun.like(Fun.lower(Get.concept_name), "%diabetes%")) |>
Select(Get.concept_id, Get.concept_name)
```

## Dates (DuckDB)

- Year of a date: `Fun.year(Get.condition_era_start_date)`, or
  `Fun.extract("year", Get.condition_era_start_date)`.
- Days between two dates: `Fun.date_diff("day", Get.start_date, Get.end_date)`.
- A date literal: `Fun.cast("2020-01-01", "DATE")`.

```julia
From(:condition_era) |>
Select(Get.person_id,
    :era_days => Fun.date_diff("day", Get.condition_era_start_date, Get.condition_era_end_date))
```

```julia
From(:visit_occurrence) |>
Group(:visit_year => Fun.year(Get.visit_start_date)) |>
Select(Get.visit_year, :n_visits => Agg.count()) |>
Order(Get.visit_year)
```

## Numeric helpers

```julia
From(:drug_exposure) |>
Group(Get.drug_concept_id) |>
Select(Get.drug_concept_id,
    :avg_days_supply => Fun.round(Agg.avg(Get.days_supply), 2),
    :max_days_supply => Agg.max(Get.days_supply))
```
