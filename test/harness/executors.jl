# Test Executors

"""
    FunctionExecutor(func::Function)

Executes a standard Julia function `func(input)`.
"""
struct FunctionExecutor <: AbstractExecutor
    func::Function
end

execute(e::FunctionExecutor, input) = e.func(input)

"""
    QueryExecutor(db_conn; dialect=FunSQL.SQLDialect(:duckdb))

Executes a SQL query string or FunSQL node on `db_conn` and returns a `DataFrame`.
"""
struct QueryExecutor <: AbstractExecutor
    db_conn::Any
    dialect::FunSQL.SQLDialect

    function QueryExecutor(db_conn; dialect=FunSQL.SQLDialect(:duckdb))
        new(db_conn, dialect)
    end
end

function execute(e::QueryExecutor, input)
    sql_str = if input isa AbstractString
        s = strip(String(input))
        if occursin("From(", s) || occursin("Select(", s) || occursin("FunSQL.", s)
            parsed = eval(Meta.parse(s))
            FunSQL.render(parsed, dialect=e.dialect)
        else
            s
        end
    elseif input isa FunSQL.SQLNode || input isa FunSQL.AbstractSQLNode
        FunSQL.render(input, dialect=e.dialect)
    else
        String(input)
    end
    res = DuckDB.execute(e.db_conn, sql_str)
    return DataFrame(res)
end

"""
    run_test_case(tc::TestCase)::TestResult

Executes `tc` using its executor (if present, otherwise passes `tc.input` directly)
and evaluates the output with `tc.validator`. Measures execution time and catches exceptions.
"""
function run_test_case(tc::TestCase)::TestResult
    t_start = time()
    actual_output = nothing
    exec_err = nothing

    try
        if tc.executor !== nothing
            actual_output = execute(tc.executor, tc.input)
        else
            actual_output = tc.input
        end
    catch err
        exec_err = err
    end

    t_elapsed = time() - t_start

    if exec_err !== nothing
        val_res = ValidationResult(
            false,
            "Execution error in test case '$(tc.name)': $(exec_err)";
            expected=tc.expected,
            actual=nothing,
            details=Dict(:error => exec_err)
        )
        return TestResult(tc, val_res; execution_time=t_elapsed, error=exec_err)
    end

    val_res = validate(tc.validator, actual_output, tc.expected)
    return TestResult(tc, val_res; execution_time=t_elapsed)
end
