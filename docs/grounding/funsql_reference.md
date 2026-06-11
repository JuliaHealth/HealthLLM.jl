# FunSQL.jl Reference

FunSQL.jl is a Julia library for composing SQL queries programmatically using function composition (`|>`).

## Core Concepts

- **Tables** are referenced with `From(:tablename)`.
- **Operations** are chained with `|>` (pipe).
- **Columns** are referenced as Julia symbols (`:column_name`).
- **Nested queries** use `From(:table) |> ...` blocks.
- **CTEs/Subqueries** use `As(:name)` to name a subquery, then `Define(:name => query)` to define it.

## Key Functions

### From
```julia
From(:person)                          # Reference a table
From(:person) |> As(:p)               # Alias a table
```

### Select
```julia
Select(:person_id, :gender_concept_id)                          # Select columns
Select(:person_id, Agg.count() |> As(:num))                     # With aggregation
Select(FunSQL.Fun("YEAR", :birth_datetime) |> As(:birth_year))  # With expressions
```

### Where / Filtering
```julia
Where(FunSQL.Fun("=", :gender_concept_id, 8507))                       # Equality
Where(FunSQL.Fun(">", :year_of_birth, 1990))                           # Comparison
Where(FunSQL.Fun("AND", FunSQL.Fun("=", :a, 1), FunSQL.Fun("=", :b, 2)))  # AND
Where(FunSQL.Fun("OR", FunSQL.Fun("=", :a, 1), FunSQL.Fun("=", :b, 2)))   # OR
Where(FunSQL.Fun("LIKE", :concept_code, "493.0%"))                     # LIKE
Where(FunSQL.Fun("IN", :concept_id, From(...) |> Select(:concept_id))) # IN subquery
Where(FunSQL.Fun("NOT IN", :concept_id, From(...) |> Select(:concept_id)))
Where(FunSQL.Fun("IS NULL", :column))                                  # IS NULL
```

### Joins
```julia
Join(From(:person), on = FunSQL.Fun("=", :care_site_id, :care_site_id))
Join(:cte_name, on = FunSQL.Fun("=", :id, :id))  # Join with a CTE/subquery
Join(From(:concept) |> As(:c), on = FunSQL.Fun("=", :c => :concept_id, :gender_concept_id))
```

### Group By and Aggregation
```julia
Group(:gender_concept_id)
Group(:gender_concept_id, :year_of_birth)
Select(:gender_concept_id, Agg.count() |> As(:cnt))
Select(:gender_concept_id, Agg.sum(:value) |> As(:total))
Select(Agg.min(:date) |> As(:first_date), Agg.max(:date) |> As(:last_date))
Select(Agg.avg(:value) |> As(:mean_value))
```

### Order By
```julia
Order(:person_id)                              # Ascending
Order(FunSQL.Fun("DESC", :person_id))          # Descending
```

### Limit / Offset
```julia
Limit(10)
Limit(10) |> Offset(20)
```

### CTEs (Common Table Expressions) with Define
```julia
From(:concept) |>
Join(From(:concept_ancestor), on = FunSQL.Fun("=", :ancestor_concept_id, :concept_id)) |>
Where(FunSQL.Fun("=", :concept_code, "73211009")) |>
Select(:descendant_concept_id |> As(:snomed_diabetes_id)) |>
As(:snomed_diabetes) |>
Define(:people_with_diabetes => From(:condition_era) |> Join(:snomed_diabetes, on = ...)) |>
From(:people_with_diabetes) |>
Select(:person_id)
```

### Distinct
```julia
From(:person) |> Select(:gender_concept_id) |> Distinct()
```

### Set Operations
```julia
From(:table1) |> Union(From(:table2))
From(:table1) |> UnionAll(From(:table2))
From(:table1) |> Intersect(From(:table2))
From(:table1) |> Except(From(:table2))
```

### Window Functions
```julia
Select(FunSQL.Fun("DENSE_RANK", FunSQL.Fun("OVER", ...)) |> As(:rank))
Select(Agg.count() |> Over(:group_id) |> As(:cnt))
```

### Date Functions
```julia
FunSQL.Fun("YEAR", :date_column)
FunSQL.Fun("MONTH", :date_column)
FunSQL.Fun("DATEDIFF", "d", :start_date, :end_date)
FunSQL.Fun("DATEADD", "d", 30, :date_column)
```

### CASE expressions
```julia
FunSQL.Fun("CASE",
    FunSQL.Fun("IN", FunSQL.Fun("MONTH", :date), 12, 1, 2), "Winter",
    FunSQL.Fun("IN", FunSQL.Fun("MONTH", :date), 3, 4, 5), "Spring",
    "Other")
```

### Literal tables (VALUES)
```julia
From([(1, "A"), (2, "B"), (3, "C")])
From([(1, "0 to 9"), (10, "10 to 19")]) |> As(:age_groups)
```

### Key Patterns

**Simple SELECT-FROM-WHERE:**
SQL: `SELECT person_id, gender_concept_id FROM person WHERE year_of_birth > 1990`
FunSQL: `From(:person) |> Where(FunSQL.Fun(">", :year_of_birth, 1990)) |> Select(:person_id, :gender_concept_id)`

**JOIN with GROUP BY:**
SQL: `SELECT cs.place_of_service_concept_id, COUNT(*) FROM care_site cs JOIN person p ON p.care_site_id = cs.care_site_id GROUP BY cs.place_of_service_concept_id`
FunSQL: `From(:care_site) |> Join(From(:person), on = FunSQL.Fun("=", :care_site_id, :care_site_id)) |> Group(:place_of_service_concept_id) |> Select(:place_of_service_concept_id, Agg.count() |> As(:cnt))`
