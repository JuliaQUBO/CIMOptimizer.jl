# CIMOptimizer.jl
[cim-optimizer](https://github.com/mcmahon-lab/cim-optimizer) Coherent Ising Machine wrapper for JuMP

[![DOI](https://zenodo.org/badge/651328206.svg)](https://zenodo.org/badge/latestdoi/651328206)
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)

## Installation
```julia
julia> import Pkg; Pkg.add(url="https://github.com/JuliaQUBO/CIMOptimizer.jl")

julia> using CIMOptimizer
```

## Getting started
```julia
using JuMP
using CIMOptimizer
const CIM = CIMOptimizer

model = Model(CIM.Optimizer)

n = 3
Q = [ -1  2  2
       2 -1  2
       2  2 -1 ]

@variable(model, x[1:n], Bin)
@objective(model, Min, x' * Q * x)

optimize!(model)

for i = 1:result_count(model)
    xi = value.(x; result = i)
    yi = objective_value(model; result = i)

    println("[$i] f($(xi)) = $(yi)")
end
```

**Note**: _The cim-optimizer wrapper for Julia is not officially supported by Cornell University's McMahon Lab. If you are interested in official support for Julia from McMachon Lab, let them know!_


**Note**: _If you are using `CoherentIsingMachine.jl` in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The [`PythonCall`](https://github.com/cjdoris/PythonCall.jl) module will place a lot of files in this folder when building its Python environment._
