using LinearAlgebra
using DelimitedFiles

function k_truss(area, modulus, length, alpha)
    c = cos(alpha)
    s = sin(alpha)
    transform = [
        c s 0.0 0.0
        -s c 0.0 0.0
        0.0 0.0 c s
        0.0 0.0 -s c
    ]
    stiffness = area * modulus / length * [
        1.0 0.0 -1.0 0.0
        0.0 0.0 0.0 0.0
        -1.0 0.0 1.0 0.0
        0.0 0.0 0.0 0.0
    ]
    return stiffness, transform
end

function write_tsv(path, data)
    mkpath(dirname(path))
    array = ndims(data) == 1 ? reshape(data, :, 1) : data
    writedlm(path, array, '\t')
end

function ex_lag_mult(outdir = "")
    A = [1.0, 1.0, 1.0]
    E = [64.0, 64.0, 64.0]
    L = [4.0, 4.0, 6.0]
    alpha = [acos(3.0 / 4.0), -acos(3.0 / 4.0), 0.0]

    Dvec = [
        1 2 5 6
        5 6 3 4
        1 2 3 4
    ]

    N = maximum(Dvec)
    Nelem, dof2 = size(Dvec)
    dof = Int(dof2 ÷ 2)
    K = zeros(Float64, N, N)

    for e in 1:Nelem
        k, T = k_truss(A[e], E[e], L[e], alpha[e])
        kg = transpose(T) * k * T
        for r in 1:(2 * dof)
            m = Dvec[e, r]
            if m != 0
                for s in 1:(2 * dof)
                    n = Dvec[e, s]
                    if n != 0
                        K[m, n] += kg[r, s]
                    end
                end
            end
        end
    end

    G = [
        1.0 0.0 0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0 0.0 0.0
        0.0 0.0 -sin(deg2rad(60.0)) cos(deg2rad(60.0)) 0.0 0.0
    ]
    Q = zeros(Float64, size(G, 1))
    a_G = maximum(K)
    Gbar = a_G * G
    Qbar = a_G * Q

    P = zeros(Float64, N)
    P[6] = -10.0

    AL = [K transpose(Gbar); Gbar zeros(Float64, size(G, 1), size(G, 1))]
    BL = vcat(P, Qbar)
    solution = AL \ BL

    U = solution[1:N]
    Lag = solution[(N + 1):end] * a_G
    R = K[1:4, :] * U

    member_forces = zeros(Float64, 4, Nelem)
    for e in 1:Nelem
        k, T = k_truss(A[e], E[e], L[e], alpha[e])
        ueg = U[Dvec[e, :]]
        ue = T * ueg
        member_forces[:, e] = k * ue
    end

    constraint_residual = G * U - Q

    if outdir != ""
        write_tsv(joinpath(outdir, "U.tsv"), U)
        write_tsv(joinpath(outdir, "Lag.tsv"), Lag)
        write_tsv(joinpath(outdir, "R.tsv"), R)
        write_tsv(joinpath(outdir, "member_forces.tsv"), member_forces)
        write_tsv(joinpath(outdir, "constraint_residual.tsv"), constraint_residual)
    else
        println("U = ", U)
        println("Lag = ", Lag)
        println("R = ", R)
        println("member_forces = ", member_forces)
    end

    return Dict(
        "U" => U,
        "Lag" => Lag,
        "R" => R,
        "member_forces" => member_forces,
        "constraint_residual" => constraint_residual,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    outdir = length(ARGS) >= 1 ? ARGS[1] : ""
    ex_lag_mult(outdir)
end
