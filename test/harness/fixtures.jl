# Fixture loaders and test dataset helpers

using JSON3
using DuckDB
using DataFrames

"""
    load_golden_cases(filepath::AbstractString)::Vector{Dict{String,Any}}

Loads golden reference cases from a JSON file.
"""
function load_golden_cases(filepath::AbstractString)::Vector{Dict{String,Any}}
    if !isfile(filepath)
        error("Golden reference file not found: $filepath")
    end
    raw = read(filepath, String)
    parsed = JSON3.read(raw, Vector{Dict{String,Any}})
    return parsed
end

"""
    save_golden_cases(filepath::AbstractString, cases)::Nothing

Saves golden reference cases to a JSON file formatted with 2-space indentation.
"""
function save_golden_cases(filepath::AbstractString, cases)::Nothing
    open(filepath, "w") do io
        JSON3.pretty(io, cases)
    end
    return nothing
end

"""
    create_test_corpus(base_dir::AbstractString)::Vector{String}

Populates `base_dir` with a hierarchy of synthetic medical text and markdown files
for testing corpus collection, concatenation, and indexing. Returns list of created file paths.
"""
function create_test_corpus(base_dir::AbstractString)::Vector{String}
    mkpath(base_dir)
    mkpath(joinpath(base_dir, "cardiology"))
    mkpath(joinpath(base_dir, "endocrinology"))
    mkpath(joinpath(base_dir, "ignored_data"))

    f1 = joinpath(base_dir, "cardiology", "hypertension_notes.txt")
    f2 = joinpath(base_dir, "cardiology", "cad_guidelines.md")
    f3 = joinpath(base_dir, "endocrinology", "diabetes_summary.txt")
    f4 = joinpath(base_dir, "ignored_data", "temp_artifact.csv")

    write(f1, "Patient presenting with stage 2 hypertension. Blood pressure 150/95 mmHg.\nPrescribed lisinopril 10mg daily.")
    write(f2, "# Coronary Artery Disease\nGuidelines for lipid management in CAD patients.\nTarget LDL < 70 mg/dL.")
    write(f3, "Type 2 Diabetes Mellitus overview.\nKey markers: HbA1c > 6.5%.\nFirst-line therapy is metformin.")
    write(f4, "id,value\n1,100\n2,200") # Non-text/csv file to verify extension filtering

    return [f1, f2, f3]
end

"""
    create_test_duckdb()::DuckDB.DB

Creates an in-memory DuckDB database preloaded with synthetic clinical tables
(`patients`, `conditions`, `observations`) for execution-based testing.
Contains ONLY non-PHI synthetic test data.
"""
function create_test_duckdb()::DuckDB.DB
    db = DuckDB.DB() # In-memory

    # Patients table
    DuckDB.execute(db, """
        CREATE TABLE patients (
            patient_id INT PRIMARY KEY,
            first_name VARCHAR,
            last_name VARCHAR,
            gender VARCHAR,
            birth_date DATE,
            age INT
        );
    """)

    DuckDB.execute(db, """
        INSERT INTO patients VALUES
            (1, 'Alice', 'Smith', 'F', '1955-03-12', 69),
            (2, 'Bob', 'Jones', 'M', '1980-07-22', 44),
            (3, 'Charlie', 'Brown', 'M', '1948-11-05', 76),
            (4, 'Diana', 'Prince', 'F', '1992-01-18', 32),
            (5, 'Evan', 'Wright', 'M', '1961-09-30', 63);
    """)

    # Conditions table
    DuckDB.execute(db, """
        CREATE TABLE conditions (
            condition_id INT PRIMARY KEY,
            patient_id INT,
            code VARCHAR,
            description VARCHAR,
            onset_date DATE
        );
    """)

    DuckDB.execute(db, """
        INSERT INTO conditions VALUES
            (101, 1, 'I10', 'Essential (primary) hypertension', '2015-05-10'),
            (102, 1, 'E11', 'Type 2 diabetes mellitus', '2018-09-14'),
            (103, 3, 'I10', 'Essential (primary) hypertension', '2010-02-20'),
            (104, 3, 'I25.1', 'Atherosclerotic heart disease', '2019-11-01'),
            (105, 5, 'E11', 'Type 2 diabetes mellitus', '2021-04-12');
    """)

    # Observations table
    DuckDB.execute(db, """
        CREATE TABLE observations (
            obs_id INT PRIMARY KEY,
            patient_id INT,
            code VARCHAR,
            description VARCHAR,
            value_numeric DOUBLE,
            units VARCHAR,
            obs_date DATE
        );
    """)

    DuckDB.execute(db, """
        INSERT INTO observations VALUES
            (201, 1, '8480-6', 'Systolic blood pressure', 145.0, 'mmHg', '2024-01-15'),
            (202, 1, '4548-4', 'Hemoglobin A1c', 7.2, '%', '2024-01-15'),
            (203, 2, '8480-6', 'Systolic blood pressure', 118.0, 'mmHg', '2024-02-10'),
            (204, 3, '8480-6', 'Systolic blood pressure', 160.0, 'mmHg', '2024-03-01'),
            (205, 5, '4548-4', 'Hemoglobin A1c', 8.1, '%', '2024-02-20');
    """)

    return db
end
