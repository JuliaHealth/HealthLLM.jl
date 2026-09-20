# Execution-based Validator for SQL / FunSQL / Generated Code

using DataFrames
using DuckDB
using FunSQL

"""
    ExecutionValidator(db_conn; catalog=nothing, dialect=FunSQL.SQLDialect(:duckdb), check_schema::Bool=true)

Validates generated SQL or FunSQL queries by executing them against a controlled test database connection `db_conn` and comparing the resulting `DataFrame` against expected results or expected criteria.

# Options
- `db_conn`: A DB connection (such as `DuckDB.DB`).
- `catalog`: Optional `FunSQL.SQLCatalog` for resolving table references during FunSQL rendering.
- `dialect`: SQL dialect for FunSQL rendering (default `:duckdb`).
- `check_schema`: Whether to verify column names match (default `true`).

# Examples
```julia
conn = DuckDB.DB() # in-memory test db
DuckDB.execute(conn, "CREATE TABLE patients (id INT, name TEXT, age INT); INSERT INTO patients VALUES (1, 'Alice', 65);")

v = ExecutionValidator(conn)
validate(v, "SELECT name FROM patients WHERE age > 60", DataFrame(name=["Alice"])) # passed
```
"""
struct ExecutionValidator <: AbstractValidator
    db_conn::Any
    catalog::Any
    dialect::FunSQL.SQLDialect
    check_schema::Bool

    function ExecutionValidator(db_conn; catalog=nothing, dialect=FunSQL.SQLDialect(:duckdb), check_schema::Bool=true)
        new(db_conn, catalog, dialect, check_schema)
    end
end

"""
    _execute_to_df(conn, query_str)::DataFrame

Helper to execute a SQL string or FunSQL rendered query on a database connection and return a DataFrame.
"""
function _execute_to_df(conn, query_str)::DataFrame
    res = DuckDB.execute(conn, string(query_str))
    return DataFrame(res)
end

function _render_funsql(v::ExecutionValidator, node)::String
    if v.catalog !== nothing
        return string(FunSQL.render(node, catalog=v.catalog, dialect=v.dialect))
    else
        return string(FunSQL.render(node, dialect=v.dialect))
    end
end

function validate(v::ExecutionValidator, actual, expected=nothing)::ValidationResult
    # 1. Resolve SQL string from actual (string or FunSQL expression/object)
    sql_to_run = ""
    try
        if actual isa AbstractString
            s = strip(String(actual))
            # Remove markdown sql codeblocks if present
            if startswith(s, "```sql") && endswith(s, "```")
                s = strip(s[7:end-3])
            elseif startswith(s, "```") && endswith(s, "```")
                s = strip(s[4:end-3])
            end

            # Check if it's FunSQL Julia code or raw SQL
            if occursin("From(", s) || occursin("Select(", s) || occursin("FunSQL.", s)
                # Parse and render FunSQL
                parsed_expr = Meta.parse(s)
                funsql_obj = eval(parsed_expr)
                sql_to_run = _render_funsql(v, funsql_obj)
            else
                sql_to_run = s
            end
        elseif actual isa FunSQL.SQLNode || actual isa FunSQL.AbstractSQLNode
            sql_to_run = _render_funsql(v, actual)
        else
            return ValidationResult(
                false,
                "ExecutionValidator expected SQL string or FunSQL node, got $(typeof(actual))";
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

    # 2. Execute SQL query on test database
    actual_df = try
        _execute_to_df(v.db_conn, sql_to_run)
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
    elseif expected isa DataFrame
        passed = isequal(actual_df, expected)
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
                "Execution result differs from expected DataFrame.\nGenerated SQL: $sql_to_run\nExpected:\n$expected\nActual:\n$actual_df";
                expected=expected,
                actual=actual_df,
                details=Dict(:sql => sql_to_run, :expected_df => expected, :actual_df => actual_df)
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
