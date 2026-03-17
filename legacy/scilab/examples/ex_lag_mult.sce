mode(-1)
// CE-515
// 22.10.2008
//
// hw2: Lagrange Multiplier method solution for constraint equations
//
// by G. Turan

function [k, T] = k_truss(A, E, L, alfa)
    T = [cos(alfa) sin(alfa) 0 0
        -sin(alfa) cos(alfa) 0 0
         0 0 cos(alfa) sin(alfa)
         0 0 -sin(alfa) cos(alfa)];
    k = A * E / L * [1 0 -1 0
                     0 0  0 0
                    -1 0  1 0
                     0 0  0 0];
endfunction

function [k, T] = k_beam(A, E, I, L, alfa)
    T = [cos(alfa) sin(alfa) 0 0 0 0
        -sin(alfa) cos(alfa) 0 0 0 0
         0 0 1 0 0 0
         0 0 0 cos(alfa) sin(alfa) 0
         0 0 0 -sin(alfa) cos(alfa) 0
         0 0 0 0 0 1];

    k = E / L * [
        A 0 0 -A 0 0
        0 12 * I / L^2 6 * I / L 0 -12 * I / L^2 6 * I / L
        0 6 * I / L 4 * I 0 -6 * I / L 2 * I
        -A 0 0 A 0 0
        0 -12 * I / L^2 -6 * I / L 0 12 * I / L^2 -6 * I / L
        0 6 * I / L 2 * I 0 -6 * I / L 4 * I
    ];
endfunction

A = [1 1 1];      // mm^2
E = [64 64 64];   // N/mm^2
L = [4 4 6];      // mm
alfa = [acos(3 / 4) -acos(3 / 4) 0];

// construct connectivity matrix
Dvec = [1 2 5 6
        5 6 3 4
        1 2 3 4];

N = max(max(Dvec));        // number of DOF in the structure
[Nelem, dof2] = size(Dvec);
dof = dof2 / 2;            // number of DOF per node

K = zeros(N, N);           // initial structural stiffness matrix

// assemble for each element
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

// Constraint equations due to boundary conditions
// G * U = Q
G = [1 0 0 0 0 0
     0 1 0 0 0 0
     0 0 -sin(60 / 180 * %pi) cos(60 / 180 * %pi) 0 0];

nc = size(G, 1);
Q = zeros(nc, 1);

// Adjust the magnitude of G so it is numerically compatible with K
a_G = max(max(K));
Gbar = a_G * G;
Qbar = a_G * Q;

P = zeros(N, 1);
P(6) = -10;  // vector of imposed loads

// Lagrange multiplier solution
AL = [K Gbar'
      Gbar zeros(nc, nc)];
BL = [P; Qbar];

solution = AL \ BL;
Lag = solution(N + 1:$);
Lag = Lag * a_G;           // backtransform the multipliers
U = solution(1:N)

// Reaction forces in global directions
R = K([1 2 3 4], :) * U

// Member forces of element 1
e = 1;
[k, T] = k_truss(A(e), E(e), L(e), alfa(e));
ueg1 = U(Dvec(e, :));
ue1 = T * ueg1;
fl1 = k * ue1

// Member forces of element 2
e = 2;
[k, T] = k_truss(A(e), E(e), L(e), alfa(e));
ueg2 = U(Dvec(e, :));
ue2 = T * ueg2;
fl2 = k * ue2

// Member forces of element 3
e = 3;
[k, T] = k_truss(A(e), E(e), L(e), alfa(e));
ueg3 = U(Dvec(e, :));
ue3 = T * ueg3;
fl3 = k * ue3
