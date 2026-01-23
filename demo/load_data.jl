using Base64
using Printf

function combine_exp_raw(root::AbstractString; output::AbstractString="combined_exp_raw.txt")
    exts = Set(lowercase.([
        ".md", ".markdown", ".html", ".rmd", ".txt", ".txtm", ".rd",
        ".csv", ".pdf", ".xlsx", ".jl"
    ]))

    open(output, "w") do out
        n = 0
        for (dirpath, _dirs, files) in walkdir(root)
            for f in files
                path = joinpath(dirpath, f)
                # get file extension (includes leading '.' or empty string)
                ext = lowercase(splitext(path)[2])
                if ext in exts
                    n += 1
                    println(out, "=== FILE: $path ===")
                    try
                        if ext in (".pdf", ".xlsx")
                            bytes = read(path)
                            b64 = base64encode(bytes)
                            println(out, "# (binary file base64-encoded)")
                            # wrap long base64 lines for readability:
                            for i in 1:76:length(b64)
                                println(out, b64[i:min(end, i+75)])
                            end
                        else
                            # treat as text
                            open(path, "r") do rf
                                for line in eachline(rf)
                                    println(out, line)
                                end
                            end
                        end
                    catch err
                        @printf(out, "# ERROR reading %s : %s\n", path, sprint(showerror, err))
                    end
                    println(out, "\n")
                end
            end
        end
        println("Wrote $n files to $(abspath(output))")
    end
end

# default invocation for the requested folder:
root = raw"E:\HealthLLM.jl\JuliaHealthLLM\data\exp_raw"
combine_exp_raw(root; output="JuliaHealthLLM_exp_raw_combined.txt")