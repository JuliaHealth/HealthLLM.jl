# SQL to FunSQL Translation Patterns

## Pattern 1: Basic SELECT-FROM
**SQL:** `SELECT col1, col2 FROM table`
**FunSQL:** `From(:table) |> Select(:col1, :col2)`

## Pattern 2: SELECT with WHERE
**SQL:** `SELECT * FROM person WHERE year_of_birth > 1990`
**FunSQL:** `From(:person) |> Where(FunSQL.Fun(">", :year_of_birth, 1990))`

## Pattern 3: JOIN
**SQL:** `SELECT * FROM a JOIN b ON a.id = b.id`
**FunSQL:** `From(:a) |> Join(From(:b), on = FunSQL.Fun("=", :id, :id))`

Note: When joining on same-named columns, the syntax is `FunSQL.Fun("=", :col, :col)`. The first `:col` is from the left side, second from the right side of the join.

**Aliased JOIN:**
**SQL:** `SELECT p.person_id, c.concept_name FROM person p JOIN concept c ON p.gender_concept_id = c.concept_id`
**FunSQL:** `From(:person) |> Join(From(:concept) |> As(:c), on = FunSQL.Fun("=", :gender_concept_id, :c => :concept_id)) |> Select(:person_id, :c => :concept_name)`

## Pattern 4: GROUP BY with Aggregation
**SQL:** `SELECT gender_concept_id, COUNT(*) AS cnt FROM person GROUP BY gender_concept_id`
**FunSQL:** `From(:person) |> Group(:gender_concept_id) |> Select(:gender_concept_id, Agg.count() |> As(:cnt))`

**SQL:** `SELECT gender_concept_id, AVG(year_of_birth) AS avg_birth FROM person GROUP BY gender_concept_id`
**FunSQL:** `From(:person) |> Group(:gender_concept_id) |> Select(:gender_concept_id, Agg.avg(:year_of_birth) |> As(:avg_birth))`

## Pattern 5: ORDER BY
**SQL:** `SELECT person_id FROM person ORDER BY person_id`
**FunSQL:** `From(:person) |> Select(:person_id) |> Order(:person_id)`

**SQL:** `SELECT person_id FROM person ORDER BY person_id DESC`
**FunSQL:** `From(:person) |> Select(:person_id) |> Order(FunSQL.Fun("DESC", :person_id))`

## Pattern 6: LIMIT / OFFSET
**SQL:** `SELECT person_id FROM person LIMIT 10 OFFSET 20`
**FunSQL:** `From(:person) |> Select(:person_id) |> Limit(10) |> Offset(20)`

## Pattern 7: DISTINCT
**SQL:** `SELECT DISTINCT gender_concept_id FROM person`
**FunSQL:** `From(:person) |> Select(:gender_concept_id) |> Distinct()`

## Pattern 8: Subquery / CTE
**SQL:**
```sql
WITH hip_fracture AS (
    SELECT DISTINCT ca.descendant_concept_id
    FROM concept c
    JOIN concept_ancestor ca ON ca.ancestor_concept_id = c.concept_id
    WHERE c.concept_code = '359817006'
)
SELECT COUNT(*) FROM condition_era ce
JOIN hip_fracture hf ON hf.descendant_concept_id = ce.condition_concept_id
```
**FunSQL:**
```julia
From(:concept) |>
Join(From(:concept_ancestor), on = FunSQL.Fun("=", :ancestor_concept_id, :concept_id)) |>
Where(FunSQL.Fun("=", :concept_code, "359817006")) |>
Select(:descendant_concept_id) |>
Distinct() |>
As(:hip_fracture) |>
Define(:result => From(:condition_era) |>
    Join(:hip_fracture, on = FunSQL.Fun("=", :descendant_concept_id, :condition_concept_id))) |>
From(:result) |>
Select(Agg.count() |> As(:cnt))
```

## Pattern 9: DATE operations
**SQL:** `SELECT YEAR(birth_datetime) FROM person`
**FunSQL:** `From(:person) |> Select(FunSQL.Fun("YEAR", :birth_datetime) |> As(:birth_year))`

**SQL:** `SELECT DATEDIFF(d, start_date, end_date) FROM table`
**FunSQL:** `From(:table) |> Select(FunSQL.Fun("DATEDIFF", "d", :start_date, :end_date) |> As(:diff))`

## Pattern 10: CASE expressions
**SQL:** `SELECT CASE WHEN MONTH(date) IN (12,1,2) THEN 'Winter' ELSE 'Other' END AS season FROM table`
**FunSQL:** `From(:table) |> Select(FunSQL.Fun("CASE", FunSQL.Fun("IN", FunSQL.Fun("MONTH", :date), 12, 1, 2), "Winter", "Other") |> As(:season))`

## Pattern 11: LIKE
**SQL:** `SELECT * FROM concept WHERE concept_code LIKE '493.0%'`
**FunSQL:** `From(:concept) |> Where(FunSQL.Fun("LIKE", :concept_code, "493.0%"))`

## Pattern 12: IN subquery
**SQL:** `SELECT * FROM condition_era WHERE condition_concept_id IN (SELECT descendant_concept_id FROM concept_ancestor WHERE ancestor_concept_id = 123)`
**FunSQL:**
```julia
From(:condition_era) |>
Where(FunSQL.Fun("IN", :condition_concept_id,
    From(:concept_ancestor) |>
    Where(FunSQL.Fun("=", :ancestor_concept_id, 123)) |>
    Select(:descendant_concept_id)))
```

## Pattern 13: UNION
**SQL:** `SELECT person_id FROM table1 UNION SELECT person_id FROM table2`
**FunSQL:** `From(:table1) |> Select(:person_id) |> Union(From(:table2) |> Select(:person_id))`

## Pattern 14: Window functions
**SQL:** `SELECT person_id, DENSE_RANK() OVER (ORDER BY year_of_birth) AS rank FROM person`
**FunSQL:** `From(:person) |> Select(:person_id, FunSQL.Fun("DENSE_RANK", FunSQL.Fun("OVER", FunSQL.Fun("ORDER", :year_of_birth))) |> As(:rank))`

## Pattern 15: Literal tables (VALUES clause)
**SQL:** `SELECT * FROM (VALUES (1, 'A'), (2, 'B')) AS t(age, grp)`
**FunSQL:** `From([(1, "A"), (2, "B")]) |> As(:t) |> Select(:column1 |> As(:age), :column2 |> As(:grp))`

## Common Translation Errors to Avoid

1. Forgetting `As(:name)` after an aggregation: `Agg.count()` must be `Agg.count() |> As(:name)`
2. Using `FROM` inside a subquery with `As()` — use `Define(name => subquery)` for CTEs
3. Column references in joins — use `table => :column` syntax for ambiguous columns, or just `:column` when unique
4. Nested FunSQL calls — `FunSQL.Fun` can be nested for complex expressions
5. String escaping — Use Julia string syntax `"value"` not SQL single quotes
