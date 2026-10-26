# CIMOptimizer.jl
[cim-optimizer](https://github.com/mcmahon-lab/cim-optimizer) Coherent Ising Machine wrapper for JuMP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8021079.svg)](https://doi.org/10.5281/zenodo.8021079)
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)

## Installation
```julia
julia> import Pkg; Pkg.add(url="https://github.com/JuliaQUBO/CIMOptimizer.jl")

julia> using CIMOptimizer
```

## Python runtime compatibility

`CIMOptimizer` uses PythonCall and CondaPkg to install `cim-optimizer ==1.0.4`
with conda `pytorch-cpu`. The CondaPkg runtime supports Python `>=3.10,<3.12`,
including Python 3.11 for shared benchmark environments with `DWave v0.7.2`.

`QiskitOpt v0.7.0` currently requires an isolated Python environment when used
alongside `CIMOptimizer`. Its PyPI `numpy ~=2.2.0` constraint conflicts with the
conda NumPy version pinned by the `pytorch-cpu` stack used here.

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

## Citation

For work that uses this Julia wrapper, cite `CIMOptimizer.jl` with its
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.8021079). The concept DOI is
the evergreen software identifier and resolves to the latest archived wrapper
release. Citation metadata is available in [`CITATION.cff`](CITATION.cff).

For exact-version reproducibility, cite the Zenodo version DOI recorded in that
GitHub release and Zenodo record. For example, the latest archived Zenodo
record under this concept is
[10.5281/zenodo.23653600](https://doi.org/10.5281/zenodo.23653600)
(`v0.2.3`).

This wrapper depends on the separately maintained upstream
[`cim-optimizer` project](https://github.com/mcmahon-lab/cim-optimizer) from
Cornell University's McMahon Lab. When using its simulator or algorithms, also
cite the upstream [`cim-optimizer` citation metadata](https://github.com/mcmahon-lab/cim-optimizer/blob/main/CITATION.cff)
and the corresponding Coherent Ising Machine literature (such as
[McMahon et al., _Science_ 2016](https://doi.org/10.1126/science.aah5178),
[Leleu et al., _Physical Review Letters_ 2019](https://doi.org/10.1103/PhysRevLett.122.040607),
and [Leleu et al., _Communications Physics_ 2021](https://doi.org/10.1038/s42005-021-00768-0)).
The `CIMOptimizer.jl` DOI identifies this Julia wrapper; it does not identify or
archive upstream `cim-optimizer`.

For general discussion or use of the JuliaQUBO ecosystem, cite the
[QUBO.jl ecosystem article](https://doi.org/10.1080/10556788.2026.2702926).

Release-time archive verification steps are documented in
[`.github/RELEASE_CHECKLIST.md`](.github/RELEASE_CHECKLIST.md).

**Note**: _The cim-optimizer wrapper for Julia is not officially supported by Cornell University's McMahon Lab. If you are interested in official support for Julia from McMachon Lab, let them know!_


**Note**: _If you are using `CoherentIsingMachine.jl` in your project, we recommend you to include the `.CondaPkg` entry in your `.gitignore` file. The [`PythonCall`](https://github.com/cjdoris/PythonCall.jl) module will place a lot of files in this folder when building its Python environment._
