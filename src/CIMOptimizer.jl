module CIMOptimizer

using PythonCall
using LinearAlgebra
using QUBODrivers: QUBODrivers, QUBOTools, MOI, Sample, SampleSet, ising

const np = PythonCall.pynew()
const co = PythonCall.pynew()
const co_si = PythonCall.pynew()
const torch = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(co, pyimport("cim_optimizer"))
    PythonCall.pycopy!(co_si, pyimport("cim_optimizer.solve_Ising"))
    PythonCall.pycopy!(torch, pyimport("torch"))
end

QUBODrivers.@setup Optimizer begin
    name = "cim-optimizer"
    sense = :min
    domain = :spin
    version = v"1.0.4" # cim-optimizer version
    attributes = begin
        "target_energy"::Number = -Inf                                  # Assumed target/ground energy for the solver to reach, used to stop before num_runs runs have been completed.
        "num_runs"::Integer = 1                                         # Maximum number of runs to attempt by the CIM, either running the set repeated number of repetitions or stopping if the target energy is met.
        "num_timesteps_per_run"::Integer = 1_000                        # Roundtrip number per run, representing the number of MVM’s per run.
        "max_wallclock_time_allowed"::Integer = 10_000_000              # Seconds passed by the CIM before quitting if num_runs or target_energy have not been met.
        "stop_when_target_energy_reached"::Bool = true                  # Stop if target energy reached before completing all the CIM runs.
        "custom_feedback_schedule"::Any = nothing                       # Option to specify a custom function or array of length num_timesteps_per_run to use as a feedback schedule.
        "custom_pump_schedule"::Any = nothing                           # Option to specify a custom function or array of length num_timesteps_per_run to use as a pump schedule.
        "hyperparameters_autotune"::Bool = false                        # If True then: Based on max_wallclock_time_allowed and num_runs, dedicate a reasonable amount of time to finding the best hyperparameters to use with the CIM.
        "hyperparameters_randomtune"::Bool = true                       # If True then: Run random hyperparameter search based on num_runs.
        "ahc_noext_time_step"::Float64 = 0.05                           # Time step for each iteration, based on Eq(X) of citation above.
        "ahc_noext_r"::Float64 = 0.2                                    # AHC hyperparameter
        "ahc_noext_beta"::Float64 = 0.05                                # AHC hyperparameter
        "ahc_noext_eps"::Float64 = 0.07                                 # AHC hyperparameter
        "ahc_noext_mu"::Float64 = 1.0                                   # AHC hyperparameter
        "ahc_noext_noise"::Float64 = 0.0                                # AHC hyperparameter
        "ahc_nonlinearity"::Any = nothing                               # Choice of amplitude control scheme to use, uses CAC for problems without external fields and AHC/CAC for problems with external fields by default.
        "cac_time_step"::Float64 = 0.05                                 # Time step for each iteration, based on Eq(X) of citation above.
        "cac_r"::Float64 = -4.04                                        # CAC hyperparameter
        "cac_alpha"::Float64 = 3.0                                      # CAC hyperparameter
        "cac_beta"::Float64 = 0.25                                      # CAC hyperparameter
        "cac_gamma"::Float64 = 0.00011                                  # CAC hyperparameter
        "cac_delta"::Float64 = 10.0                                     # CAC hyperparameter
        "cac_mu"::Float64 = 1.0                                         # CAC hyperparameter
        "cac_rho"::Float64 = 3.0                                        # CAC hyperparameter
        "cac_tau"::Float64 = 1_000.0                                    # CAC hyperparameter
        "cac_noise"::Float64 = 0.0                                      # CAC hyperparameter
        "cac_nonlinearity"::Any = nothing # np.tanh                     # CAC hyperparameter
        "ahc_ext_time_step"::Integer = 9_000                            # Roundtrip number per run, representing time horizon.
        "ahc_ext_nsub"::Integer = 1                                     # extAHC hyperparameter
        "ahc_ext_alpha"::Float64 = 1.0                                  # extAHC hyperparameter
        "ahc_ext_delta"::Float64 = 0.25                                 # extAHC hyperparameter
        "ahc_ext_eps"::Float64 = 0.333                                  # extAHC hyperparameter
        "ahc_ext_lambd"::Float64 = 0.001                                # extAHC hyperparameter
        "ahc_ext_pi"::Float64 = -0.225                                  # extAHC hyperparameter
        "ahc_ext_rho"::Float64 = 1.0                                    # extAHC hyperparameter
        "ahc_ext_tau"::Float64 = 100.0                                  # extAHC hyperparameter
        "ahc_ext_F_h"::Float64 = 2.0                                    # extAHC hyperparameter
        "ahc_ext_noise"::Float64 = 0.0                                  # extAHC hyperparameter
        "ahc_ext_nonlinearity"::Any = nothing # torch.tanh              # extAHC hyperparameter
        "return_lowest_energies_found_spin_configuration"::Bool = false # Return a vector where for each run, it gives the spin configuration that was found during that run that had the lowest energy.
        "return_lowest_energy_found_from_each_run"::Bool = true         # Return a vector with the lowest energy found for each run.
        "return_spin_trajectories_all_runs"::Bool = true                # Return the Ising spin trajectory for every run.
        "return_number_of_solutions"::Integer = 1_000                   # Number of best solutions to return; must be <= num_runs.
        "suppress_statements"::Bool = false                             # Print details of each solver call to screen/REPL.
        "use_GPU"::Bool = false                                         # Option to use GPU acceleration using PyTroch libraries.
        "use_CAC"::Bool = true                                          # Option to select CAC or AHC solver for no external field.
        "chosen_device"::Any = nothing # torch.device("cpu")            # Device used for torch-based computations.
    end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    h, J, α, β = ising(sampler, Matrix)

    # cim-optimzier asks for symmetric 'J', and its convention
    # is to min -s'J s -h's
    h = np.array(-h)
    J = np.array(-Symmetric(J))

    model = co_si.Ising(J, h)

    # Retrieve attributes
    num_runs = MOI.get(sampler, MOI.RawOptimizerAttribute("num_runs"))

    if MOI.get(sampler, MOI.Silent())
        MOI.set(sampler, MOI.RawOptimizerAttribute("suppress_statements"), true)
    end

    if MOI.get(sampler, MOI.TimeLimitSec()) !== nothing
        MOI.get(sampler, MOI.RawOptimizerAttribute("max_wallclock_time_allowed"), MOI.get(sampler, MOI.TimeLimitSec()))
    end

    solver = model.solve(;
        target_energy                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("target_energy")),
        num_runs                                        = MOI.get(sampler, MOI.RawOptimizerAttribute("num_runs")),
        num_timesteps_per_run                           = MOI.get(sampler, MOI.RawOptimizerAttribute("num_timesteps_per_run")),
        max_wallclock_time_allowed                      = MOI.get(sampler, MOI.RawOptimizerAttribute("max_wallclock_time_allowed")),
        stop_when_target_energy_reached                 = MOI.get(sampler, MOI.RawOptimizerAttribute("stop_when_target_energy_reached")),
        custom_feedback_schedule                        = MOI.get(sampler, MOI.RawOptimizerAttribute("custom_feedback_schedule")),
        custom_pump_schedule                            = MOI.get(sampler, MOI.RawOptimizerAttribute("custom_pump_schedule")),
        hyperparameters_autotune                        = MOI.get(sampler, MOI.RawOptimizerAttribute("hyperparameters_autotune")),
        hyperparameters_randomtune                      = MOI.get(sampler, MOI.RawOptimizerAttribute("hyperparameters_randomtune")),
        ahc_noext_time_step                             = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_noext_time_step")),
        ahc_noext_r                                     = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_noext_r")),
        ahc_noext_beta                                  = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_noext_beta")),
        ahc_noext_eps                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_noext_eps")),
        ahc_noext_mu                                    = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_noext_mu")),
        ahc_noext_noise                                 = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_noext_noise")),
        ahc_nonlinearity                                = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_nonlinearity")),
        cac_time_step                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_time_step")),
        cac_r                                           = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_r")),
        cac_alpha                                       = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_alpha")),
        cac_beta                                        = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_beta")),
        cac_gamma                                       = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_gamma")),
        cac_delta                                       = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_delta")),
        cac_mu                                          = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_mu")),
        cac_rho                                         = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_rho")),
        cac_tau                                         = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_tau")),
        cac_noise                                       = MOI.get(sampler, MOI.RawOptimizerAttribute("cac_noise")),
        cac_nonlinearity                                = something(MOI.get(sampler, MOI.RawOptimizerAttribute("cac_nonlinearity")), np.tanh),
        ahc_ext_time_step                               = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_time_step")),
        ahc_ext_nsub                                    = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_nsub")),
        ahc_ext_alpha                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_alpha")),
        ahc_ext_delta                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_delta")),
        ahc_ext_eps                                     = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_eps")),
        ahc_ext_lambd                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_lambd")),
        ahc_ext_pi                                      = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_pi")),
        ahc_ext_rho                                     = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_rho")),
        ahc_ext_tau                                     = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_tau")),
        ahc_ext_F_h                                     = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_F_h")),
        ahc_ext_noise                                   = MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_noise")),
        ahc_ext_nonlinearity                            = something(MOI.get(sampler, MOI.RawOptimizerAttribute("ahc_ext_nonlinearity")), torch.tanh),
        return_lowest_energies_found_spin_configuration = MOI.get(sampler, MOI.RawOptimizerAttribute("return_lowest_energies_found_spin_configuration")),
        return_lowest_energy_found_from_each_run        = MOI.get(sampler, MOI.RawOptimizerAttribute("return_lowest_energy_found_from_each_run")),
        return_spin_trajectories_all_runs               = MOI.get(sampler, MOI.RawOptimizerAttribute("return_spin_trajectories_all_runs")),
        return_number_of_solutions                      = MOI.get(sampler, MOI.RawOptimizerAttribute("return_number_of_solutions")),
        suppress_statements                             = MOI.get(sampler, MOI.RawOptimizerAttribute("suppress_statements")),
        use_GPU                                         = MOI.get(sampler, MOI.RawOptimizerAttribute("use_GPU")),
        use_CAC                                         = MOI.get(sampler, MOI.RawOptimizerAttribute("use_CAC")),
        chosen_device                                   = something(MOI.get(sampler, MOI.RawOptimizerAttribute("chosen_device")), torch.device("cpu")),
    )

    samples = Vector{Sample{T,Int}}(undef, num_runs)

    for i = 1:num_runs
        ψ = round.(Int, pyconvert.(T, solver.result["spin_config_all_runs"][i-1]))
        λ = α * (pyconvert(T, solver.result["energies"][i-1]) + β)

        samples[i] = Sample{T}(ψ, λ)
    end

    metadata = Dict{String,Any}(
        "origin" => "cim-optimizer",
        "time"   => Dict{String,Any}(
            "effective" => solver.result["time"]
        ),
    )

    return SampleSet{T}(samples, metadata)
end

end # module CoherentIsingMachine
