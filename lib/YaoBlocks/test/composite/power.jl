using Test, YaoBlocks
using LinearAlgebra

# Multi-qubit circuits to use as Power content
circ2 = chain(2, put(1 => Rx(0.3)), put(2 => Rz(0.7)), cnot(2, 1, 2))
circ3 = chain(3, put(1 => H), kron(3, 1 => Rx(0.2), 2 => Ry(0.4)), put(3 => T))

@testset "power constructor" begin
    pb = power(X, 3)
    @test pb isa Power
    @test pb == Power(X, 3)

    pb = X ^ 3
    @test pb isa Power
    @test pb == Power(X, 3)
end

@testset "mat correctness" begin
    for (gate, nq_gate) in [(X, 1), (Rx(0.3), 1), (circ2, 2), (circ3, 3)]
        for n in [-2, 0, 1, 2, 3]
            pb = power(gate, n)
            # Use Matrix() to avoid sparse-type issues with ^0
            @test Matrix(mat(pb)) ≈ Matrix(mat(gate))^n
            @test Matrix(mat(pb')) ≈ Matrix(mat(gate'))^n
            @test Matrix(mat(pb')) ≈ Matrix(mat(pb))'
        end
    end
end

@testset "apply! on random state" begin
    for (gate, nq_gate) in [(X, 1), (Rx(0.3), 1), (circ2, 2), (circ3, 3)]
        for n in [0, 1, 2, 3]
            pb = power(gate, n)
            reg = rand_state(nq_gate)
            reg_ref = copy(reg)
            for _ in 1:n
                apply!(reg_ref, gate)
            end
            apply!(reg, pb)
            @test statevec(reg) ≈ statevec(reg_ref)

            # adjoint
            reg2 = rand_state(nq_gate)
            reg2_ref = copy(reg2)
            for _ in 1:n
                apply!(reg2_ref, gate')
            end
            apply!(reg2, pb')
            @test statevec(reg2) ≈ statevec(reg2_ref)
        end
    end
end

@testset "negative powers" begin
    # Negative power requires unitary content
    @test_throws ArgumentError power(matblock(rand(ComplexF64, 2, 2)), -1)

    for (gate, nq_gate) in [(X, 1), (Rx(0.3), 1), (circ2, 2), (circ3, 3)]
        for n in [-1, -2, -3]
            pb = power(gate, n)
            # apply!: should be the same as applying adjoint |n| times
            reg = rand_state(nq_gate)
            reg_ref = copy(reg)
            for _ in 1:abs(n)
                apply!(reg_ref, gate')
            end
            apply!(reg, pb)
            @test statevec(reg) ≈ statevec(reg_ref)
            # adjoint of a negative-power block
            @test Matrix(mat(pb')) ≈ Matrix(mat(pb))'
        end
    end
end

nq = 4
@testset "inside put block" begin
    for (gate, nq_gate) in [(X, 1), (Rx(0.3), 1), (circ2, 2), (circ3, 3)]
        locs = Tuple(1:nq_gate)
        for n in [0, 1, 2, 3]
            pb = power(gate, n)
            circuit = put(nq, locs => pb)
            reg = rand_state(nq)
            reg_ref = copy(reg)
            for _ in 1:n
                apply!(reg_ref, put(nq, locs => gate))
            end
            apply!(reg, circuit)
            @test statevec(reg) ≈ statevec(reg_ref)
            @test Matrix(mat(circuit)) ≈ Matrix(mat(put(nq, locs => gate)))^n
        end
    end
end

@testset "inside control block" begin
    for (gate, nq_gate) in [(X, 1), (Rx(0.3), 1), (circ2, 2)]
        ctrl_loc = nq_gate + 1
        locs = Tuple(1:nq_gate)
        for n in [1, 2, 3]
            pb = power(gate, n)
            nq_circ = nq_gate + 1
            circuit = control(nq_circ, ctrl_loc, locs => pb)
            reg = rand_state(nq_circ)
            reg_ref = copy(reg)
            for _ in 1:n
                apply!(reg_ref, control(nq_circ, ctrl_loc, locs => gate))
            end
            apply!(reg, circuit)
            @test statevec(reg) ≈ statevec(reg_ref)
        end
    end
end
