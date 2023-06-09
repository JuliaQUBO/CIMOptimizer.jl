using CoherentIsingMachine: CoherentIsingMachine, MOI, QUBODrivers

QUBODrivers.test(CoherentIsingMachine.Optimizer; examples=true) do model
    MOI.set(model, MOI.Silent(), true)
end
