using Base64

function combine_ohdsi_funsql(ohdsi_file::String, funsql_file::String, output_file::String)
    open(output_file, "w") do out
        # Write OHDSI part
        println(out, "# OHDSI Data")
        open(ohdsi_file, "r") do f
            for line in eachline(f)
                println(out, line)
            end
        end
        println(out, "\n")

        # Write FunSQL part
        println(out, "# FunSQL Data")
        open(funsql_file, "r") do f
            for line in eachline(f)
                println(out, line)
            end
        end
        println(out, "\n")
    end
    println("Combined data written to $(abspath(output_file))")
end

# Files
ohdsi_file = "demo/JuliaHealthLLM_exp_raw_combined.txt"
funsql_file = "FunSQL_combined.txt"
output_file = "OHDSI_FunSQL_combined.txt"

combine_ohdsi_funsql(ohdsi_file, funsql_file, output_file)