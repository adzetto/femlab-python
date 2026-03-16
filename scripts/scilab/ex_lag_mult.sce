mode(0);
ieee(1);

function [k, T] = k_truss(A, E, L, alfa)
    T = [cos(alfa) sin(alfa) 0 0
        -sin(alfa) cos(alfa) 0 0
         0 0 cos(alfa) sin(alfa)
         0 0 -sin(alfa) cos(alfa)];
    k = A * E / L * [1 0 -1 0
                     0 0 0 0
                    -1 0 1 0
                     0 0 0 0];
endfunction

function write_tsv(path, data)
    sep = ascii(9);
    if size(data, "*") == 0 then
        mputl("", path);
        return
    end
    csvWrite(data, path, sep);
endfunction

outdir = getenv("EX_LAG_MULT_OUTDIR");

A = [1 1 1];
E = [64 64 64];
L = [4 4 6];
alfa = [acos(3 / 4) -acos(3 / 4) 0];

Dvec = [1 2 5 6
        5 6 3 4
        1 2 3 4];

N = max(max(Dvec));
[Nelem, dof2] = size(Dvec);
dof = dof2 / 2;
K = zeros(N, N);

for e = 1:Nelem
    [k, T] = k_truss(A(e), E(e), L(e), alfa(e));
    kg = T' * k * T;
    for r = 1:2 * dof
        m = Dvec(e, r);
        if m <> 0 then
            for s = 1:2 * dof
                n = Dvec(e, s);
                if n <> 0 then
                    K(m, n) = K(m, n) + kg(r, s);
                end
            end
        end
    end
end

G = [1 0 0 0 0 0
     0 1 0 0 0 0
     0 0 -sin(60 / 180 * %pi) cos(60 / 180 * %pi) 0 0];
nc = size(G, 1);
Q = zeros(nc, 1);
a_G = max(max(K));
Gbar = a_G * G;
Qbar = a_G * Q;

P = zeros(N, 1);
P(6) = -10;

AL = [K Gbar'
      Gbar zeros(nc, nc)];
BL = [P; Qbar];
solution = AL \ BL;

Lag = solution(N + 1:$) * a_G;
U = solution(1:N);
R = K([1 2 3 4], :) * U;

member_forces = zeros(4, Nelem);
for e = 1:Nelem
    [k, T] = k_truss(A(e), E(e), L(e), alfa(e));
    ueg = U(Dvec(e, :));
    ue = T * ueg;
    member_forces(:, e) = k * ue;
end

constraint_residual = G * U - Q;

if outdir <> "" then
    if ~isdir(outdir) then
        mkdir(outdir);
    end
    write_tsv(outdir + "/U.tsv", U);
    write_tsv(outdir + "/Lag.tsv", Lag);
    write_tsv(outdir + "/R.tsv", R);
    write_tsv(outdir + "/member_forces.tsv", member_forces);
    write_tsv(outdir + "/constraint_residual.tsv", constraint_residual);
else
    disp("U =");
    disp(U);
    disp("Lag =");
    disp(Lag);
    disp("R =");
    disp(R);
    disp("member_forces =");
    disp(member_forces);
end

exit;
