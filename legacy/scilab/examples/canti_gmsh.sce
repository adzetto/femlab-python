
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

G_=load_gmsh("/home/gursoy/Dersler/ce-512/meshing/deneme4.msh");

// Topology for 4-node element    
//     Node1 Node2 Node3 Node4    PropNo
T = [G_.QUADS(:,1:4) G_.ELE_INFOS(:,5)];

// Node coordinates  
//      X      Y
X = G_.POS(:,1:2);

// No. of dof per node (Q4E element)  
dof = 2;

// Material properties (Elastic element)
//       E     nu   type = plane stress
G = [100,0.3,1
     200,0.3,1];

// Constrained dof 
//    NodeNo   DofNo   U
C = [1,1,0;
     7,1,0;
     7,2,0;
     2,1,0];

// Nodal loads 
//    NodeNo    Px    Py
P = [3,0,-0.05;
     10,0,-0.1;
     4,0,-0.05];
