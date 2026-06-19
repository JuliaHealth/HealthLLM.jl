@echo off
set JULIA_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=1
julia --project --heap-size-hint=4G -t 2 %*
