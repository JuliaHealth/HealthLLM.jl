# DuckDB plumbing for the FunSQL benchmark: expose an OMOP CDM DuckDB file under
# the `cdm` schema the gold SQL expects, reflect a FunSQL catalog from it, and run
# queries with a wall-clock timeout.

using DuckDB
using DataFrames
using FunSQL
using HealthLLM: ResultSet

"""
    open_omop_duckdb(path; source_schema="dbt_synthea_dev") -> (; conn, catalog, tables)

Open the OMOP DuckDB database at `path` read-only and make its tables available
as `cdm.<table>`, which the gold SQL in the benchmark uses. The search path is
set to `cdm` too, so the unqualified table names that FunSQL renders also resolve.

The source file is attached `READ_ONLY` into a fresh in-memory database and the
`cdm` schema holds only views, so a benchmark run can never modify it, whatever
SQL a model generates. Only base tables are exposed. The Synthea ETL file also
contains staging views that point at schemas it does not ship.

`catalog` is a `FunSQL.SQLCatalog` reflected from the `cdm` schema. Rendering
generated FunSQL against it rejects any table or column the database lacks.
"""
function open_omop_duckdb(path::AbstractString; source_schema::AbstractString="dbt_synthea_dev")
    isfile(path) || error("OMOP DuckDB file not found: $path")
    db = DuckDB.DB()
    conn = DBInterface.connect(db)
    # An in-memory database has nowhere to spill, so large joins and sorts over
    # the full patient table fail with "Out of Memory". Give it a temp directory.
    DBInterface.execute(conn, "SET temp_directory = '$(replace(mktempdir(), '\\' => '/'))'")
    DBInterface.execute(conn, "SET preserve_insertion_order = false")
    DBInterface.execute(conn, "ATTACH '$(replace(path, "'" => "''"))' AS omop_src (READ_ONLY)")
    DBInterface.execute(conn, "CREATE SCHEMA cdm")
    tables = DataFrame(DBInterface.execute(conn, """
        SELECT table_name FROM information_schema.tables
        WHERE table_catalog = 'omop_src' AND table_schema = '$source_schema'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name""")).table_name
    isempty(tables) && error("No tables found in schema `$source_schema` of $path")
    for t in tables
        DBInterface.execute(conn,
            "CREATE VIEW cdm.\"$t\" AS SELECT * FROM omop_src.\"$source_schema\".\"$t\"")
    end
    DBInterface.execute(conn, "SET search_path = 'cdm'")
    catalog = FunSQL.reflect(conn; schema="cdm", dialect=:duckdb)
    return (; conn, catalog, tables=String.(tables))
end

struct QueryTimeout <: Exception
    seconds::Float64
end
Base.showerror(io::IO, e::QueryTimeout) = print(io, "query exceeded the $(e.seconds)s timeout")

"""
    execute_with_timeout(conn, sql; timeout=60) -> DataFrame

Run `sql` on `conn` and return the result, interrupting it after `timeout` seconds.

DuckDB.jl's own `execute` blocks until the query finishes, and a watchdog task
cannot interrupt it. Generated queries do sometimes produce runaway joins over
the full patient table. So this drives DuckDB's pending-query API itself: it
executes one task at a time and checks the deadline in between. This uses
DuckDB.jl internals (`Stmt`, `PendingQueryResult`), verified against DuckDB.jl 1.5.
"""
function execute_with_timeout(conn, sql::AbstractString; timeout::Real=60)
    stmt = DuckDB.Stmt(conn, sql, DuckDB.MaterializedResult)
    pending = DuckDB.PendingQueryResult(stmt)
    pending.success || throw(DuckDB.QueryException(DuckDB.get_error(stmt, pending)))
    deadline = time() + timeout
    interrupted = false
    state = DuckDB.DUCKDB_PENDING_RESULT_NOT_READY
    while !DuckDB.duckdb_pending_execution_is_finished(state)
        if !interrupted && time() > deadline
            DuckDB.duckdb_interrupt(conn.handle)
            interrupted = true
        end
        state = DuckDB.duckdb_pending_execute_task(pending.handle)
    end
    interrupted && throw(QueryTimeout(Float64(timeout)))
    state == DuckDB.DUCKDB_PENDING_ERROR &&
        throw(DuckDB.QueryException(DuckDB.get_error(stmt, pending)))
    handle = Ref{DuckDB.duckdb_result}()
    if DuckDB.duckdb_execute_pending(pending.handle, handle) != DuckDB.DuckDBSuccess
        msg = DuckDB.fetch_error(sql, DuckDB.duckdb_result_error(handle))
        DuckDB.duckdb_destroy_result(handle)
        throw(DuckDB.QueryException(msg))
    end
    return DataFrame(DuckDB.QueryResult(handle))
end

"""
    resultset(df) -> ResultSet

Convert a DataFrame to the backend-neutral `ResultSet` the harness compares.
"""
resultset(df::DataFrame) = ResultSet(names(df), [collect(Any, r) for r in eachrow(df)])

"""
    duckdb_executor(conn; timeout=60, max_rows=200_000) -> (sql -> ResultSet)

The `executor` the harness calls for both gold and generated SQL. A result
larger than `max_rows` throws, because comparing it would cost more than the
task is worth. In practice, results that large come from a missing filter or
aggregate.
"""
function duckdb_executor(conn; timeout::Real=60, max_rows::Integer=200_000)
    return function (sql)
        df = execute_with_timeout(conn, sql; timeout=timeout)
        nrow(df) > max_rows && error("result has $(nrow(df)) rows (limit $max_rows)")
        return resultset(df)
    end
end
