using CIMOptimizer: CIMOptimizer, MOI, QUBODrivers

QUBODrivers.test(CIMOptimizer.Optimizer; examples=true) do model
    MOI.set(model, MOI.Silent(), true)
end
