function [K] = kh8e(K,T,X,G)

//***************************************************
// kh8e: 
//   Creates and assembles stiffness matrix for
//   a group of elastic quadrilateral 4-node 
//   elements in plane stress or plane strain.
// Syntax:
//   K = kh8e(K,T,X,G)
// Input:
//   K  :  initial global stiffness matrix.
//   T  :  element topology matrix.
//   X  :  node coordinate matrix. 
//   G  :  material property matrix. 
// Output:
//   K  :  new global stiffness matrix.
// Date:
//   Version 1.0    13.04.2007
//***************************************************

// determine number of nodes per element  
nnodes = cols(T)-1;


for j = 1:rows(T)
   // define element arrays
  Xe = X(T(j,1:nnodes),:);
  Ge = G(T(j,nnodes+1),:);

  // evaluate element stiffness
  Ke = keh8e(Xe,Ge);

  // assemble element stiffness into global stiffness
  K = assmk(K,Ke,T(j,:),3);
end

endfunction
