# Execution-based Validator for SQL / FunSQL / Generated Code

using DataFrames
using DuckDB
using FunSQL
using HealthLLM
using HealthLLM.Evaluation: ResultSet, results_match
using HealthLLM.Execution: sanity_check_funsql, FunSQLGeneration, FunSQLCheck

"""
    ExecutionValidator(db_conn; catalog=nothing, dialect=:duckdb, check_schema::Bool=true, ordered::Bool=false, atol::Real=1e-6, rtol::Real=1e-4)

Validates generated SQL or FunSQL queries by executing them against a controlled test database connection or executor function `db_conn` and comparing the resulting `DataFrame` or `ResultSet` against expected results or criteria.

Reuses `HealthLLM.Execution.sanity_check_funsql` for sandboxed FunSQL parsing and rendering, and `HealthLLM.Evaluation.results_match` for semantic result set comparison.

# Options
- `db_conn`: A DB connection (such as `DuckDB.DB`) or an executor function `sql -> DataFrame` / `sql -> ResultSet`.
- `catalog`: Optional `FunSQL.SQLCatalog` for resolving table references during FunSQL rendering.
- `dialect`: SQL dialect for FunSQL rendering (default `:duckdb`).
- `check_schema`: Whether to verify column names match (default `true`).
- `ordered`: Whether row ordering matters (default `false`).
- `atol`, `rtol`: Tolerances for numerical comparison in `results_match`.
"""
struct ExecutionValidator <: AbstractValidator
    db_conn::Any
    catalog::Any
    dialect::Any
    check_schema::Bool
    ordered::Bool
    atol::Float64
    rtol::Float64

    function ExecutionValidator(
        db_conn;
        catalog=nothing,
        dialect=:duckdb,
        check_schema::Bool=true,
        ordered::Bool=false,
        atol::Real=1e-6,
        rtol::Real=1e-4
    )
        new(db_conn, catalog, dialect, check_schema, ordered, Float64(atol), Float64(rtol))
    end
end

_to_resultset(rs::ResultSet) = rs
_to_resultset(df::DataFrame) = ResultSet(names(df), [collect(Any, r) for r in eachrow(df)])

function _to_dataframe(rs::ResultSet)
    if isempty(rs.rows)
        return DataFrame([name => Any[] for name in rs.columns])
    end
    cols = [getindex.(rs.rows, i) for i in 1:length(rs.columns)]
    return DataFrame(cols, rs.columns)
end

function _execute_query(conn, sql_str::AbstractString)
    if conn isa Function
        res = conn(sql_str)
        if res isa DataFrame
            return res
        elseif res isa ResultSet
            return _to_dataframe(res)
        else
            return DataFrame(res)
        end
    else
        res = DuckDB.execute(conn, string(sql_str))
        return DataFrame(res)
    end
end

function validate(v::ExecutionValidator, actual, expected=nothing)::ValidationResult
    dialect_sym = v.dialect isa FunSQL.SQLDialect ? :duckdb : v.dialect

    # 1. Resolve SQL string from actual (string, FunSQLGeneration, or FunSQL node)
    sql_to_run = ""
    try
        if actual isa FunSQLGeneration
            check = sanity_check_funsql(actual; catalog=v.catalog, dialect=dialect_sym)
            if !check.ok
                return ValidationResult(
                    false,
                    "Failed to parse/render query at stage $(check.stage): $(check.error)";
                    actual=actual,
                    details=Dict(:error => check.error, :stage => check.stage)
                )
            end
            sql_to_run = check.sql
        elseif actual isa AbstractString
            s = strip(String(actual))
            # Remove markdown sql/julia codeblocks if present
            if startswith(s, "```sql") && endswith(s, "```")
                s = strip(s[7:end-3])
            elseif startswith(s, "```julia") && endswith(s, "```")
                s = strip(s[9:end-3])
            elseif startswith(s, "```") && endswith(s, "```")
                s = strip(s[4:end-3])
            end

            # Check if it's FunSQL Julia code or raw SQL
            if occursin("From(", s) || occursin("Select(", s) || occursin("FunSQL.", s) || occursin("|>", s)
                check = sanity_check_funsql(s; catalog=v.catalog, dialect=dialect_sym)
                if !check.ok
                    return ValidationResult(
                        false,
                        "Failed to parse/render query at stage $(check.stage): $(check.error)";
                        actual=actual,
                        details=Dict(:error => check.error, :stage => check.stage)
                    )
                end
                sql_to_run = check.sql
            else
                sql_to_run = s
            end
        elseif actual isa FunSQL.SQLNode || actual isa FunSQL.AbstractSQLNode
            if v.catalog !== nothing
                sql_to_run = string(FunSQL.render(v.catalog, actual))
            elseif v.dialect isa FunSQL.SQLDialect
                sql_to_run = string(FunSQL.render(actual; dialect=v.dialect))
            else
                sql_to_run = string(FunSQL.render(actual; dialect=dialect_sym))
            end
        else
            return ValidationResult(
                false,
                "ExecutionValidator expected SQL string, FunSQLGeneration, or FunSQL node, got $(typeof(actual))";
                actual=actual
            )
        end
    catch err
        return ValidationResult(
            false,
            "Failed to parse/render query: $err";
            actual=actual,
            details=Dict(:error => err)
        )
    end

    # 2. Execute SQL query on test database or via executor
    actual_df = try
        _execute_query(v.db_conn, sql_to_run)
    catch err
        return ValidationResult(
            false,
            "Execution error while running generated query:\nQuery: $sql_to_run\nError: $err";
            expected=expected,
            actual=actual,
            details=Dict(:sql => sql_to_run, :execution_error => err)
        )
    end

    # 3. Compare with expected
    if expected === nothing
        # Just validating successful execution
        return ValidationResult(
            true,
            "Query executed successfully, returned $(nrow(actual_df)) rows";
            actual=actual_df,
            details=Dict(:sql => sql_to_run, :row_count => nrow(actual_df))
        )
    elseif expected isa DataFrame || expected isa ResultSet
        expected_rs = _to_resultset(expected)
        actual_rs = _to_resultset(actual_df)

        schema_ok = true
        if v.check_schema
            gnames = lowercase.(expected_rs.columns)
            pnames = lowercase.(actual_rs.columns)
            schema_ok = (length(gnames) == length(pnames)) && (Set(gnames) == Set(pnames))
        end

        passed = schema_ok && results_match(expected_rs, actual_rs; ordered=v.ordered, atol=v.atol, rtol=v.rtol)
        if passed
            return ValidationResult(
                true,
                "Execution result matches expected DataFrame";
                expected=expected,
                actual=actual_df,
                details=Dict(:sql => sql_to_run)
            )
        else
            return ValidationResult(
                false,
                "Execution result differs from expected $(expected isa DataFrame ? "DataFrame" : "ResultSet").\nGenerated SQL: $sql_to_run\nExpected:\n$expected\nActual:\n$actual_df";
                expected=expected,
                actual=actual_df,
                details=Dict(:sql => sql_to_run, :expected => expected, :actual_df => actual_df)
            )
        end
    elseif expected isa Function
        # Custom predicate function on DataFrame: expected(df) -> Bool
        passed = try
            expected(actual_df)
        catch err
            return ValidationResult(
                false,
                "Predicate validation error: $err";
                actual=actual_df,
                details=Dict(:sql => sql_to_run)
            )
        end
        msg = passed ? "Execution predicate passed" : "Execution predicate failed on result DataFrame"
        return ValidationResult(passed, msg; actual=actual_df, details=Dict(:sql => sql_to_run))
    elseif expected isa Integer
        # Expected row count
        actual_rows = nrow(actual_df)
        passed = actual_rows == expected
        msg = passed ? "Row count matches ($expected rows)" : "Expected $expected rows, got $actual_rows rows"
        return ValidationResult(passed, msg; expected=expected, actual=actual_rows, details=Dict(:sql => sql_to_run))
    else
        return ValidationResult(
            false,
            "Unsupported expected type for ExecutionValidator: $(typeof(expected))";
            expected=expected,
            actual=actual_df
        )
    end
end
