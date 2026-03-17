#=
CE 512 HW3 - Large-Scale Rayleigh-Ritz: n x n System Scaling
Multi-threaded BLAS (all CPU cores) for dense assembly + solve.
n up to 10000 to demonstrate:
  1. O(n^2) assembly, O(n^3) solve scaling
  2. Exponential condition-number growth (Hilbert-like K)
  3. Float64 breakdown around n ~ 20-25

Outputs data_scaling.dat for pgfplots.
Normalized: P=1, EA=1, L=1.
=#

using LinearAlgebra, Printf

# Use all available BLAS threads
nt = Sys.CPU_THREADS
BLAS.set_num_threads(nt)
println("BLAS threads: $nt")

const OUTDIR = @__DIR__

# --- Exact solution (normalized) ---
u_exact(x) = x <= 0.5 ? 2x : 0.5 + x

# --- In-place assembly of K and F ---
function assemble!(K, F, n)
    @inbounds for b in 1:n
        j = b + 1
        for a in 1:n
            i = a + 1
            K[a, b] = (i * j) / (i + j - 1)
        end
        F[b] = 1.0 / 2.0^(b + 1) + 1.0
    end
end

# --- Evaluate u(x) from coefficients ---
function eval_u(x, n, c)
    u = 0.0
    xp = x * x  # start at x^2
    @inbounds for a in 1:n
        u += c[a] * xp
        xp *= x
    end
    return u
end

# --- Evaluate sigma(x) = du/dx from coefficients ---
function eval_sigma(x, n, c)
    s = 0.0
    xp = x  # start at x^1 for d/dx(x^2)=2x
    @inbounds for a in 1:n
        k = a + 1
        s += c[a] * k * xp
        xp *= x
    end
    return s
end

# ===================== CONFIGURATION =====================
n_vals = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50,
          75, 100, 150, 200, 300, 500, 750, 1000,
          1500, 2000, 3000, 5000, 7500, 10000]

# Limits for expensive computations
const COND_MAX_N  = 2000   # cond() via SVD up to this n
const ERROR_MAX_N = 500    # L2 error via quadrature up to this n

# ===================== MAIN LOOP =====================
println("=" ^ 70)
println("CE 512 HW3 - Large-Scale Rayleigh-Ritz Scaling Study")
println("=" ^ 70)
println()

open(joinpath(OUTDIR, "data_scaling.dat"), "w") do io
    println(io, "n\tasm_s\tsolve_s\tcond_K\tresid\tu_tip\tu_tip_err\tu_L2\tsigma_L2")

    for n in n_vals
        @printf("n = %5d ... ", n)

        K = Matrix{Float64}(undef, n, n)
        F = Vector{Float64}(undef, n)

        # --- Assembly ---
        t_asm = @elapsed assemble!(K, F, n)

        # --- Condition number ---
        if n <= COND_MAX_N
            kn = cond(Symmetric(K))
        else
            kn = NaN
        end

        # --- Solve (Cholesky for SPD, fallback to LU) ---
        local c
        t_solve = @elapsed begin
            try
                c = cholesky(Symmetric(K)) \ F
            catch
                try
                    c = lu(K) \ F
                catch
                    c = fill(NaN, n)
                end
            end
        end

        # --- Relative residual ---
        resid = norm(K * c - F) / norm(F)

        # --- Tip displacement ---
        u_tip = eval_u(1.0, n, c)
        tip_err = abs(u_tip - 1.5) / 1.5 * 100

        # --- L2 errors (only for moderate n where evaluation is stable) ---
        if n <= ERROR_MAX_N && isfinite(u_tip) && abs(u_tip) < 1e10
            x_q = range(0, 1, length=2000)
            dx = x_q[2] - x_q[1]
            du2 = 0.0; ds2 = 0.0
            for x in x_q
                ue = u_exact(x)
                se = x < 0.5 ? 2.0 : (x > 0.5 ? 1.0 : 1.5)
                ur = eval_u(x, n, c)
                sr = eval_sigma(x, n, c)
                du2 += (ur - ue)^2 * dx
                ds2 += (sr - se)^2 * dx
            end
            u_l2 = sqrt(abs(du2))
            s_l2 = sqrt(abs(ds2))
        else
            u_l2 = NaN
            s_l2 = NaN
        end

        @printf(io, "%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n",
                n, t_asm, t_solve, kn, resid, u_tip, tip_err, u_l2, s_l2)

        @printf("asm=%.4fs  solve=%.4fs  cond=%.2e  resid=%.2e  u(L)=%.6f\n",
                t_asm, t_solve, kn, resid, u_tip)
    end
end

println()
println("File written: data_scaling.dat")
println("Done.")
