
// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//.........................................
// File:canti.sce
// 
// Example : Bending of cantilever beam. 
// Element Type: Q4E 
// No. of Nodes: 27
// No. of Elements: 16
//......................................... 
dir="~/Dersler/ce-512/FemLab";

chdir(dir);

a=load_gmsh("mesh/deneme.msh");

// Topology for 4-node element    
//     Node1 Node2 Node3 Node4    PropNo
elno=a.TRIANGLES(:,4);

T = [ a.TRIANGLES(:,1:3) a.ELE_INFOS(elno,5) ];

// Node coordinates  
//      X      Y
X = a.POS(:,1:2);

// No. of dof per node (Q4E element)  
dof = 2;

// Material properties (Elastic element)
//       E     nu   type = plane stress
G = [2e8,0.3,1
     0.7e8, 0.23, 1];

// Constrained dof 
//    NodeNo   DofNo   U
C = [5,1,0;
     7,1,0;
     8,1,0;
     8, 2, 0
     9, 2, 0
     10,2,0
     9,1,0
     10, 1, 0
     11, 1, 0
     6, 1, 0];

// Nodal loads 
//    NodeNo    Px    Py
P = [25,0,-0.05;
     24,0,-0.1;
     22,0,-0.05];
     
     
[K,p,q] = init(rows(X),dof);

// Assembly of element stiffness matrices.
K = kt3e(K,T,X,G);
// Set nodal loads:
p = setload(p,P);

// Solve the system with Lagrange Multipliers
u = solve_lag(K,p,C,dof);

// Postprocessing: 
//  q: internal force vector
//  S: stress matrix
//  E: strain matrix
[q,S,E] = qt3e(q,T,X,G,u);


// Open figure window No. 1
scf(1);
clf()
// Plot elements and nodes of topology T. 
plotelem(T,X)

set(gca(),"auto_clear","off")

plotforces(T,X,P)
plotbc(T,X,C)

// Plot displaced configuration X+U (scaled)
U = matrix(u,size(X'))';
plotelem(T,X+1000*U,"c:")
set(gca(),"auto_clear","on")

// Open figure window No. 2
scf(2); clf()
// Contour plot of stress component No. 1: Sx
plott3(T,X,S,1)

// extract reaction forces
R = reaction(q,C,dof);
