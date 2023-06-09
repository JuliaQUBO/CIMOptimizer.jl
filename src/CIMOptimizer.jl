module CIMOptimizer

using PythonCall
using LinearAlgebra
using QUBODrivers: QUBODrivers, QUBOTools, MOI, Sample, SampleSet, ising

const np = PythonCall.pynew()
const co = PythonCall.pynew()
const co_si = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(co, pyimport("cim_optimizer"))
    PythonCall.pycopy!(co_si, pyimport("cim_optimizer.solve_Ising"))
end

QUBODrivers.@setup Optimizer begin
    name = "cim-optimizer"
    sense = :min
    domain = :spin
    version = v"1.0.4" # cim-optimizer version
    attributes = begin
        NumberOfRuns["num_runs"]::Integer = 1
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
    num_runs = MOI.get(sampler, NumberOfRuns())

    solver = model.solve(;
        num_runs            = num_runs,
        suppress_statements = MOI.get(sampler, MOI.Silent()),
    )

    samples = Vector{Sample{T,Int}}(undef, num_runs)

    for i = 1:num_runs
        ψ = round.(Int, pyconvert.(T, solver.result["spin_config_all_runs"][i-1]))
        λ = α * (pyconvert(T, solver.result["energies"][i-1]) + β)

        samples[i] = Sample{T}(ψ, λ)
    end

    metadata = Dict{String,Any}(
        "origin" => "cim-optimizer",
        "time" => Dict{String,Any}(
            "effective" => solver.result["time"]
        ),
    )

    return SampleSet{T}(samples, metadata)
end

end # module CoherentIsingMachine
