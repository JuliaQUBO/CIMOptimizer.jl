using Pkg
using TOML
using Test

using CIMOptimizer: CIMOptimizer, MOI, QUBODrivers

@testset "Dependencies" begin
    @test !any(dep -> dep.name == "Anneal", values(Pkg.dependencies()))

    conda = TOML.parsefile(joinpath(pkgdir(CIMOptimizer), "CondaPkg.toml"))
    pip_deps = get(get(conda, "pip", Dict{String,Any}()), "deps", Dict{String,Any}())

    @test haskey(conda["deps"], "pytorch-cpu")
    @test !haskey(pip_deps, "torch")
end

@testset "Julia support policy" begin
    project = TOML.parsefile(joinpath(pkgdir(CIMOptimizer), "Project.toml"))
    compat = project["compat"]

    @test compat["julia"] == "1.10"
    @test compat["QUBODrivers"] == "0.4, 0.5"

    ci = read(joinpath(pkgdir(CIMOptimizer), ".github", "workflows", "ci.yml"), String)
    ci_versions = Set(m.captures[1] for m in eachmatch(r"version:\s*'([^']+)'", ci))

    @test compat["julia"] in ci_versions
    @test "1" in ci_versions
end

@testset "QUBODrivers interface" begin
    QUBODrivers.test(CIMOptimizer.Optimizer; examples=true) do model
        MOI.set(model, MOI.Silent(), true)
        MOI.set(model, MOI.RawOptimizerAttribute("hyperparameters_randomtune"), false)
        MOI.set(model, MOI.RawOptimizerAttribute("num_timesteps_per_run"), 10)
        MOI.set(model, MOI.RawOptimizerAttribute("return_number_of_solutions"), 1)
        MOI.set(model, MOI.RawOptimizerAttribute("return_spin_trajectories_all_runs"), false)
    end
end
