function K = kT4e(K,T,X,G)
// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// kT4E: 
//   Creates and assembles stiffness matrix for
//   a group of elastic triangular 3-node
//   elements in plane stress or plane strain.
// Syntax:
//   K = kT4e(K,T,X,G)
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

// Generate and assemble 4 noded tetrahedra elements (T4e).
for j = 1:rows(T)

  // define element arrays
  Xe = X(T(j,1:4),:);
  Ge = G(T(j,5),:);

  // evaluate element stiffness
  Ke = keT4e(Xe,Ge);

  // assemble element stiffness into global stiffness
  K  = assmk(K,Ke,T(j,:),3);
end

endfunction
