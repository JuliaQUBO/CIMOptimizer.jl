using Pkg
using TOML
using Test

using CIMOptimizer: CIMOptimizer, MOI, QUBODrivers

@testset "Package metadata" begin
    project = TOML.parsefile(joinpath(pkgdir(CIMOptimizer), "Project.toml"))

    @test "David E. Bernal Neira <dbernaln@purdue.edu>" in project["authors"]
end

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
    @test compat["QUBODrivers"] == "0.6.1"

    ci = read(joinpath(pkgdir(CIMOptimizer), ".github", "workflows", "ci.yml"), String)
    normalized_ci = replace(ci, "\r\n" => "\n")
    ci_versions = Set(m.captures[1] for m in eachmatch(r"version:\s*'([^']+)'", ci))

    @test compat["julia"] in ci_versions
    @test "1" in ci_versions

    setup = findfirst("julia-actions/setup-julia@v3", ci)
    cache = findfirst("julia-actions/cache@v3", ci)
    buildpkg = findfirst("julia-actions/julia-buildpkg@v1", ci)

    @test setup !== nothing
    @test cache !== nothing
    @test buildpkg !== nothing
    @test last(setup) < first(cache) < first(buildpkg)
    @test occursin(r"(?m)^permissions:\n\s+actions:\s*write\n\s+contents:\s*read", normalized_ci)
end

@testset "Dependency maintenance policy" begin
    config = read(joinpath(pkgdir(CIMOptimizer), ".github", "dependabot.yml"), String)
    normalized = replace(config, r"\s+" => " ")
    workflows = readdir(joinpath(pkgdir(CIMOptimizer), ".github", "workflows"))

    @test occursin("version: 2", config)
    @test occursin("""package-ecosystem: "julia" directory: "/" """, normalized)
    @test occursin("""package-ecosystem: "github-actions" directory: "/" """, normalized)
    @test occursin("root-julia-dependencies", config)
    @test !any(workflow -> occursin("compathelper", lowercase(workflow)), workflows)
end

@testset "Typed attributes and capabilities" begin
    sampler = CIMOptimizer.Optimizer{Float64}()

    @test MOI.supports(sampler, CIMOptimizer.NumberOfRuns())
    @test MOI.supports(sampler, CIMOptimizer.MaxWallclockTimeAllowed())
    @test MOI.supports(sampler, CIMOptimizer.ReturnNumberOfSolutions())
    @test MOI.supports(sampler, QUBODrivers.FinalNumberOfReads())
    @test !MOI.supports(sampler, QUBODrivers.RandomSeed())

    MOI.set(sampler, CIMOptimizer.NumberOfRuns(), 3)
    MOI.set(sampler, CIMOptimizer.MaxWallclockTimeAllowed(), 2.5)
    MOI.set(sampler, CIMOptimizer.ReturnNumberOfSolutions(), 2)
    MOI.set(sampler, QUBODrivers.FinalNumberOfReads(), 2)

    @test MOI.get(sampler, CIMOptimizer.NumberOfRuns()) == 3
    @test MOI.get(sampler, MOI.RawOptimizerAttribute("num_runs")) == 3
    @test MOI.get(sampler, CIMOptimizer.MaxWallclockTimeAllowed()) == 2.5
    @test MOI.get(sampler, MOI.RawOptimizerAttribute("max_wallclock_time_allowed")) == 2.5
    @test MOI.get(sampler, CIMOptimizer.ReturnNumberOfSolutions()) == 2
    @test MOI.get(sampler, MOI.RawOptimizerAttribute("return_number_of_solutions")) == 2
    @test MOI.get(sampler, QUBODrivers.FinalNumberOfReads()) == 2

    @test !QUBODrivers.supports_seed(CIMOptimizer.Optimizer)
    @test QUBODrivers.honors_final_reads(CIMOptimizer.Optimizer)
    @test !QUBODrivers.enforces_time_limit(CIMOptimizer.Optimizer)
end

@testset "QUBODrivers interface" begin
    QUBODrivers.test(CIMOptimizer.Optimizer; examples=true, benchmark_conformance=true) do model
        MOI.set(model, MOI.Silent(), true)
        MOI.set(model, MOI.RawOptimizerAttribute("hyperparameters_randomtune"), false)
        MOI.set(model, MOI.RawOptimizerAttribute("num_timesteps_per_run"), 10)
        MOI.set(model, MOI.RawOptimizerAttribute("return_number_of_solutions"), 1)
        MOI.set(model, MOI.RawOptimizerAttribute("return_spin_trajectories_all_runs"), false)
    end
end
