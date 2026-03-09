
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
g=load_gmsh("mesh/deneme.msh");

// Topology for 4-node element    
//     Node1 Node2 Node3 Node4    PropNo
T = g.QUADS(:,1:4);
T = [T g.ELE_INFOS(:,5)];

// Node coordinates  
//      X      Y
X = g.POS(:,1:2);

// No. of dof per node (Q4E element)  
dof = 2;

// Material properties (Elastic element)
//       E     nu   type = plane stress
G = [100,0.3,1];

// Constrained dof 
//    NodeNo   DofNo   U
C = [1,1,0;
     2,1,0;
     2,2,0;
     3,1,0];

// Nodal loads 
//    NodeNo    Px    Py
P = [25,0,-0.05;
     26,0,-0.1;
     27,0,-0.05];
