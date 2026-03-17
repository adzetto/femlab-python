function [K] = assmk(K,Ke,Te,dof)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);





//***************************************************
// assmk: 
//   Assembles system matrix by adding element 
//   matrix to existing global matrix. The system
//   matrix may be a stiffness matrix, mass matrix,
//   conductivity matrix, etc. 
// Syntax:
//   K = assmk(K,Ke,Te)
//   K = assmk(K,Ke,Te,dof)
// Input:
//   K  :  global matrix.
//   Ke :  element matrix.
//   Te :  element topology vector.
//   dof:  degrees of freedom per node.
// Output:
//   K  :  updated global matrix. 
// Date:
//   Version 1.0    04.05.95
//***************************************************

// Default number of dof = 1
if %nargin==3 then
  dof = 1;
end;

// Set number of element nodes.
ne = size(Te,2)-1;

// Define global address vector ig( ) for element dofs.
ig = zeros(1,ne*dof);
for i = 1:ne
  for j = 1:dof
    ig((i-1)*dof+j) = (Te(i)-1)*dof + j;
  end;
end;

// Add element contributions.
for i = 1:ne*dof
  for j = 1:ne*dof

    K(ig(i),ig(j)) = K(ig(i),ig(j))+Ke(i,j);
  end;
end;
endfunction
