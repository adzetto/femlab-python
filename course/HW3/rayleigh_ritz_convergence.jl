#=
CE 512 HW3 - Rayleigh-Ritz Convergence Study
General n-term polynomial: u(x) = a_2 x^2 + a_3 x^3 + ... + a_{n+1} x^{n+1}
(a_1 x term omitted as in the homework)

Outputs .dat files for pgfplots import in LaTeX.
Normalized: P=1, EA=1, L=1
=#

using LinearAlgebra, Printf

const OUTDIR = @__DIR__

# --- Exact solution (normalized) ---
u_exact(x) = x <= 0.5 ? 2x : 0.5 + x
sigma_exact(x) = x < 0.5 ? 2.0 : (x > 0.5 ? 1.0 : 1.5)

# --- Rayleigh-Ritz solver for n basis functions {x^2, ..., x^{n+1}} ---
function solve_rr(n)
    ks = 2:n+1
    K = zeros(n, n)
    for (a, i) in enumerate(ks), (b, j) in enumerate(ks)
        K[a, b] = i * j / (i + j - 1)
    end
    F = [1.0 / 2.0^i + 1.0 for i in ks]
    return collect(ks), K \ F
end

# --- Evaluate displacement and stress ---
function eval_rr(x, ks, a)
    u = sum(a[m] * x^ks[m] for m in eachindex(ks))
    s = sum(a[m] * ks[m] * x^(ks[m]-1) for m in eachindex(ks))
    return u, s
end

# ==================== CONFIGURATION ====================
x_fine = range(0, 1, length=500)
n_values = [1, 2, 3, 4, 5, 8, 12, 20, 30, 50]

# ==================== DISPLACEMENT DATA ====================
# Header: x  exact  n1  n2  n3  ...
open(joinpath(OUTDIR, "data_displacement.dat"), "w") do io
    # header
    print(io, "x\texact")
    for n in n_values
        print(io, "\tn$n")
    end
    println(io)
    # precompute solutions
    sols = [solve_rr(n) for n in n_values]
    for x in x_fine
        @printf(io, "%.6f\t%.6f", x, u_exact(x))
        for (ks, a) in sols
            u, _ = eval_rr(x, ks, a)
            @printf(io, "\t%.6f", u)
        end
        println(io)
    end
end

# ==================== STRESS DATA ====================
open(joinpath(OUTDIR, "data_stress.dat"), "w") do io
    print(io, "x\texact")
    for n in n_values
        print(io, "\tn$n")
    end
    println(io)
    sols = [solve_rr(n) for n in n_values]
    for x in x_fine
        se = x < 0.5 ? 2.0 : (x > 0.5 ? 1.0 : 1.5)
        @printf(io, "%.6f\t%.6f", x, se)
        for (ks, a) in sols
            _, s = eval_rr(x, ks, a)
            @printf(io, "\t%.6f", s)
        end
        println(io)
    end
end

# ==================== ERROR DATA ====================
n_range = 1:60
open(joinpath(OUTDIR, "data_errors.dat"), "w") do io
    println(io, "n\tu_L2\tu_Linf\tsigma_L2\tsigma_Linf\tu_tip_pct")
    x_quad = range(0, 1, length=2000)
    dx = x_quad[2] - x_quad[1]
    for n in n_range
        ks, a = solve_rr(n)
        du2 = 0.0; du_max = 0.0
        ds2 = 0.0; ds_max = 0.0
        for x in x_quad
            u_rr, s_rr = eval_rr(x, ks, a)
            ue = u_exact(x)
            se = x < 0.5 ? 2.0 : (x > 0.5 ? 1.0 : 1.5)
            du = abs(u_rr - ue); ds = abs(s_rr - se)
            du2 += du^2 * dx; ds2 += ds^2 * dx
            du_max = max(du_max, du); ds_max = max(ds_max, ds)
        end
        u_tip = eval_rr(1.0, ks, a)[1]
        tip_pct = abs(u_tip - 1.5) / 1.5 * 100
        @printf(io, "%d\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n",
                n, sqrt(du2), du_max, sqrt(ds2), ds_max, tip_pct)
    end
end

# ==================== PRINT SUMMARY ====================
println("=" ^ 65)
println("CE 512 HW3 - Rayleigh-Ritz Convergence (data export)")
println("=" ^ 65)
println()
@printf("%-5s  %10s  %10s  %10s  %10s\n",
        "n", "u(L)", "u_err(%)", "||u||_L2", "||s||_L2")
println("-" ^ 50)

for n in n_values
    ks, a = solve_rr(n)
    u_tip = eval_rr(1.0, ks, a)[1]
    # quick L2
    dx2 = 1.0/2000; d2 = 0.0
    for x in range(0,1,length=2000)
        du = abs(eval_rr(x,ks,a)[1] - u_exact(x))
        d2 += du^2 * dx2
    end
    @printf("%-5d  %10.6f  %10.4f  %10.6f\n",
            n, u_tip, abs(u_tip-1.5)/1.5*100, sqrt(d2))
end

println()
println("Files written:")
println("  data_displacement.dat")
println("  data_stress.dat")
println("  data_errors.dat")
