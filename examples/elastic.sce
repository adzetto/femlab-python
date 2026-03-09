
// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//.........................................
// File:elastic.m 
// 
// Driver for linear analysis of elastic 
// problem with Q4E elements. 
// 
// Required input:
//  T:   Topology
//  X:   Node coordinates
//  G:   Material properties
//  C:   Prescribed displacements
//  P:   Prescribed nodal loads
//  dof: No. of dof per node
//.........................................   

// Initialization:  
//   K: Global stiffness matrix 
//   p: Global load vector
//   q: Global internal force vector 
[K,p,q] = init(size(X,1),dof);

// Assembly of element stiffness matrices.
K = kq4e(K,T,X,G);
// Set nodal loads:
p = setload(p,P);

// Solve the system with Lagrange Multipliers
//  u = solve_lag(K,p,C,dof);

// Set boundary conditions. 
[K,p] = setbc(K,p,C,dof);

// Solvev equations:
//  u: displacement vector

u = inv(K)*p;

// Postprocessing: 
//  q: internal force vector
//  S: stress matrix
//  E: strain matrix
[q,S,E] = qq4e(q,T,X,G,u);

// Open figure window No. 1
scf(1);
clf()

// Plot elements and nodes of topology T. 
plotelem(T,X)

plotforces(T,X,P)
plotbc(T,X,C)

// Plot displaced configuration X+U (scaled)
U = matrix(u,size(X'))';

plotelem(T,X+ 1*U,"c:")

// Open figure window No. 2
scf(2);
clf()

// Contour plot of stress component No. 1: Sx
plotq4(T,X,S,1)

// extract reaction forces
R = reaction(q,C,dof);

