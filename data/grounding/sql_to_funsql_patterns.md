# SQL to FunSQL Translation Patterns

Each pattern pairs a SQL query with an equivalent FunSQL.jl query over the OMOP
CDM. The FunSQL side is one Julia expression and renders to SQL with
`FunSQL.render(catalog, query)`.

## Pattern 1: SELECT with WHERE

SQL:

```sql
SELECT person_id, gender_concept_id FROM person WHERE year_of_birth > 1990
```

FunSQL:

```julia
From(:person) |>
Where(Get.year_of_birth .> 1990) |>
Select(Get.person_id, Get.gender_concept_id)
```

## Pattern 2: COUNT of all rows

SQL:

```sql
SELECT COUNT(*) AS n_persons FROM person
```

FunSQL:

```julia
From(:person) |>
Group() |>
Select(:n_persons => Agg.count())
```

## Pattern 3: COUNT DISTINCT

SQL:

```sql
SELECT COUNT(DISTINCT person_id) AS n_patients FROM condition_occurrence
```

FunSQL:

```julia
From(:condition_occurrence) |>
Group() |>
Select(:n_patients => Agg.count_distinct(Get.person_id))
```

## Pattern 4: GROUP BY with ORDER BY

SQL:

```sql
SELECT gender_concept_id, COUNT(*) AS n_persons
FROM person
GROUP BY gender_concept_id
ORDER BY n_persons DESC
```

FunSQL:

```julia
From(:person) |>
Group(Get.gender_concept_id) |>
Select(Get.gender_concept_id, :n_persons => Agg.count()) |>
Order(Get.n_persons |> Desc())
```

## Pattern 5: HAVING

SQL:

```sql
SELECT person_id, COUNT(*) AS n_drugs
FROM drug_era
GROUP BY person_id
HAVING COUNT(*) >= 3
```

FunSQL:

```julia
From(:drug_era) |>
Group(Get.person_id) |>
Where(Agg.count() .>= 3) |>
Select(Get.person_id, :n_drugs => Agg.count())
```

## Pattern 6: SELECT DISTINCT

SQL:

```sql
SELECT DISTINCT drug_concept_id FROM drug_era
```

FunSQL:

```julia
From(:drug_era) |>
Group(Get.drug_concept_id) |>
Select(Get.drug_concept_id)
```

## Pattern 7: JOIN to look up a concept name

SQL:

```sql
SELECT c.concept_name AS gender, COUNT(*) AS n_persons
FROM person p
JOIN concept c ON p.gender_concept_id = c.concept_id
GROUP BY c.concept_name
```

FunSQL:

```julia
From(:person) |>
Join(:c => From(:concept), Get.gender_concept_id .== Get.c.concept_id) |>
Group(Get.c.concept_name) |>
Select(:gender => Get.concept_name, :n_persons => Agg.count())
```

## Pattern 8: JOIN two clinical tables on person_id

SQL:

```sql
SELECT cs.place_of_service_concept_id, COUNT(*) AS n_persons
FROM care_site cs
JOIN person p ON p.care_site_id = cs.care_site_id
GROUP BY cs.place_of_service_concept_id
```

FunSQL:

```julia
From(:care_site) |>
Join(:p => From(:person), Get.care_site_id .== Get.p.care_site_id) |>
Group(Get.place_of_service_concept_id) |>
Select(Get.place_of_service_concept_id, :n_persons => Agg.count())
```

## Pattern 9: All descendants of a concept (concept_ancestor)

OMOP stores clinical hierarchies in `concept_ancestor`. To find every record of a
condition *including its sub-types*, match on the descendants of the ancestor
concept rather than on the concept id itself.

SQL:

```sql
SELECT COUNT(DISTINCT ce.person_id) AS n_patients
FROM condition_era ce
JOIN concept_ancestor ca ON ce.condition_concept_id = ca.descendant_concept_id
WHERE ca.ancestor_concept_id = 201826
```

FunSQL:

```julia
From(:condition_era) |>
Join(:ca => From(:concept_ancestor), Get.condition_concept_id .== Get.ca.descendant_concept_id) |>
Where(Get.ca.ancestor_concept_id .== 201826) |>
Group() |>
Select(:n_patients => Agg.count_distinct(Get.person_id))
```

## Pattern 10: Find a concept by source code

SQL:

```sql
SELECT concept_id, concept_name
FROM concept
WHERE vocabulary_id = 'SNOMED' AND concept_code = '44054006'
```

FunSQL:

```julia
From(:concept) |>
Where((Get.vocabulary_id .== "SNOMED") .& (Get.concept_code .== "44054006")) |>
Select(Get.concept_id, Get.concept_name)
```

## Pattern 11: IN subquery

SQL:

```sql
SELECT COUNT(*) AS n_exposures
FROM drug_exposure
WHERE person_id IN (SELECT person_id FROM death)
```

FunSQL:

```julia
From(:drug_exposure) |>
Where(Fun.in(Get.person_id, From(:death) |> Select(Get.person_id))) |>
Group() |>
Select(:n_exposures => Agg.count())
```

## Pattern 12: CTE (WITH)

SQL:

```sql
WITH elderly AS (
    SELECT person_id FROM person WHERE year_of_birth < 1950
)
SELECT COUNT(*) AS n_visits
FROM visit_occurrence v
JOIN elderly e ON v.person_id = e.person_id
```

FunSQL:

```julia
From(:visit_occurrence) |>
Join(:e => From(:elderly), Get.person_id .== Get.e.person_id) |>
Group() |>
Select(:n_visits => Agg.count()) |>
With(:elderly => From(:person) |> Where(Get.year_of_birth .< 1950) |> Select(Get.person_id))
```

## Pattern 13: Average of a per-person count

SQL:

```sql
SELECT AVG(n_eras) AS avg_eras
FROM (SELECT person_id, COUNT(*) AS n_eras FROM condition_era GROUP BY person_id) t
```

FunSQL:

```julia
From(:condition_era) |>
Group(Get.person_id) |>
Select(Get.person_id, :n_eras => Agg.count()) |>
Group() |>
Select(:avg_eras => Agg.avg(Get.n_eras))
```

## Pattern 14: Duration in days

SQL (DuckDB):

```sql
SELECT person_id, date_diff('day', drug_era_start_date, drug_era_end_date) AS era_days
FROM drug_era
```

FunSQL:

```julia
From(:drug_era) |>
Select(Get.person_id,
    :era_days => Fun.date_diff("day", Get.drug_era_start_date, Get.drug_era_end_date))
```

## Pattern 15: Group by year

SQL (DuckDB):

```sql
SELECT year(condition_era_start_date) AS era_year, COUNT(*) AS n_eras
FROM condition_era
GROUP BY era_year
ORDER BY era_year
```

FunSQL:

```julia
From(:condition_era) |>
Group(:era_year => Fun.year(Get.condition_era_start_date)) |>
Select(Get.era_year, :n_eras => Agg.count()) |>
Order(Get.era_year)
```

## Pattern 16: Age at an event

SQL:

```sql
SELECT co.person_id, year(co.condition_start_date) - p.year_of_birth AS age_at_onset
FROM condition_occurrence co
JOIN person p ON co.person_id = p.person_id
```

FunSQL:

```julia
From(:condition_occurrence) |>
Join(:p => From(:person), Get.person_id .== Get.p.person_id) |>
Select(Get.person_id,
    :age_at_onset => Fun.year(Get.condition_start_date) .- Get.p.year_of_birth)
```

## Pattern 17: Top N

SQL:

```sql
SELECT drug_concept_id, COUNT(*) AS n_exposures
FROM drug_exposure
GROUP BY drug_concept_id
ORDER BY n_exposures DESC
LIMIT 10
```

FunSQL:

```julia
From(:drug_exposure) |>
Group(Get.drug_concept_id) |>
Select(Get.drug_concept_id, :n_exposures => Agg.count()) |>
Order(Get.n_exposures |> Desc()) |>
Limit(10)
```

## Common translation errors

1. Using `Distinct()`, `Offset()`, `Union()`, or `Having()`. These don't exist in
   FunSQL. Use `Group`, `Limit(a:b)`, `Append`, and `Where` after `Group`.
2. An aggregate without `Group`. `Agg.count()` is only valid after `Group(...)`
   (or `Partition(...)`). For a whole-table count, use `Group()`.
3. An unqualified join condition. Write `Get.person_id .== Get.p.person_id`, not
   `Fun("=", :person_id, :person_id)`, which compares a column with itself.
4. SQL Server date functions. `DATEDIFF(DAY, a, b)` and `DATEADD` are not DuckDB.
   Use `Fun.date_diff("day", a, b)` and date arithmetic.
5. Using `==` instead of `.==`. Comparisons inside a query must use the broadcast
   forms `.==`, `.>`, `.&`, and so on.
6. Single-quoted strings. Julia strings use double quotes: `"SNOMED"`.
