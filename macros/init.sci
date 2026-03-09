function [K,p,q] = init(nn,dof)

// Ouput variables initialisation (not found in input variables)
K=[];
p=[];
q=[];

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// init:
//   Initializes global stiffness matrix, global
//   load vector and global internal force vector.
// Syntax:
//   K = init(nn,dof)
//   [K,p] = init(nn,dof)
//   [K,p,q] = init(nn,dof)
// Input:
//   nn  :  number of global nodes.
//   dof :  number of degrees of freedom per node.
// Output:
//   K   :  initialized global stiffness matrix.
//   p   :  initialized global load vector.
//   q   :  initialized global internal force vector.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// square matrix K(nn*dof,nn*dof)
if nn<1000
   K = zeros(nn*dof,nn*dof);
else
   // format K as a sparse matrix
   K = spzeros(nn*dof,nn*dof);
end

// column vector p(nn*dof)
if %nargout>1 then
  p = zeros(nn*dof,1);
end;

// column vector q(nn*dof)
if %nargout==3 then
  q = zeros(nn*dof,1);
end;
endfunction
