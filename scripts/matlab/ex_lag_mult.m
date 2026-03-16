function result = ex_lag_mult(output_dir)
if nargin < 1
    output_dir = "";
end

A = [1 1 1];
E = [64 64 64];
L = [4 4 6];
alpha = [acos(3 / 4), -acos(3 / 4), 0];

Dvec = [
    1 2 5 6
    5 6 3 4
    1 2 3 4
];

N = max(Dvec, [], "all");
[Nelem, dof2] = size(Dvec);
dof = dof2 / 2;
K = zeros(N, N);

for e = 1:Nelem
    [k, T] = k_truss(A(e), E(e), L(e), alpha(e));
    kg = T' * k * T;
    for r = 1:(2 * dof)
        m = Dvec(e, r);
        if m ~= 0
            for s = 1:(2 * dof)
                n = Dvec(e, s);
                if n ~= 0
                    K(m, n) = K(m, n) + kg(r, s);
                end
            end
        end
    end
end

G = [
    1 0 0 0 0 0
    0 1 0 0 0 0
    0 0 -sind(60) cosd(60) 0 0
];
Q = zeros(size(G, 1), 1);
a_G = max(K, [], "all");
Gbar = a_G * G;
Qbar = a_G * Q;

P = zeros(N, 1);
P(6) = -10;

AL = [K, Gbar'; Gbar, zeros(size(G, 1), size(G, 1))];
BL = [P; Qbar];
solution = AL \ BL;

Lag = solution((N + 1):end) * a_G;
U = solution(1:N);
R = K(1:4, :) * U;

member_forces = zeros(4, Nelem);
for e = 1:Nelem
    [k, T] = k_truss(A(e), E(e), L(e), alpha(e));
    ueg = U(Dvec(e, :));
    ue = T * ueg;
    member_forces(:, e) = k * ue;
end

constraint_residual = G * U - Q;

result = struct( ...
    "U", U, ...
    "Lag", Lag, ...
    "R", R, ...
    "member_forces", member_forces, ...
    "constraint_residual", constraint_residual ...
);

if strlength(output_dir) > 0
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    write_tsv(fullfile(output_dir, "U.tsv"), U);
    write_tsv(fullfile(output_dir, "Lag.tsv"), Lag);
    write_tsv(fullfile(output_dir, "R.tsv"), R);
    write_tsv(fullfile(output_dir, "member_forces.tsv"), member_forces);
    write_tsv(fullfile(output_dir, "constraint_residual.tsv"), constraint_residual);
elseif nargout == 0
    disp(result);
end
end

function [k, T] = k_truss(A, E, L, alpha)
c = cos(alpha);
s = sin(alpha);
T = [
    c s 0 0
    -s c 0 0
    0 0 c s
    0 0 -s c
];
k = A * E / L * [
    1 0 -1 0
    0 0 0 0
    -1 0 1 0
    0 0 0 0
];
end

function write_tsv(path, data)
writematrix(data, path, "Delimiter", "tab", "FileType", "text");
end
