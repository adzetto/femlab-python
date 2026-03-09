function [K] = kq4epe(K,T,X,G,S,E,mtype)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// KQ4EPE: 
//   Creates and assembles tangent stiffness matrix
//   for a group of elasto-plastic quadrilateral
//   4-node elements in plane strain.
// Syntax:
//   K = kq4epe(K,T,X,G,S,E)
//   K = kq4epe(K,T,X,G,S,E,type)
// Input:
//   K    : initial global stiffness matrix.
//   T    : element topology matrix.
//   X    : node coordinate matrix. 
//   G    : material property matrix. 
//   S    : current stress matrix.
//   E    : current strain matrix.
//   mtype : parameter setting material model
//          mtype = 1 -> Von Mises (default)
//          mtype = 2 -> Drucker-Prager
// Output:
//   K    : new global stiffness matrix.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// set default material model to Von Mises
if %nargin==6 then
  mtype = 1;
end;

// if not defined - expand S 

if (cols(S)~=20) then
  S(1,20) = 0;
end;
// if not defined - expand E 

if (cols(E)~=20) then
  E(1,20) = 0;
end;

for j = 1:rows(T)
  // extract element information from global arrays
  Xe = X(T(j,1:4),:);
  Ge = G(T(j,5),:);
  // select row j and reshape into element format
  Se = matrix(S(j,:),5,4)';
  Ee = matrix(E(j,:),5,4)';

  // evaluate element stiffness
  Ke = keq4epe(Xe,Ge,Se,Ee,mtype);

  // assemble element stiffness in global stiffness
  K = assmk(K,Ke,T(j,:),2);
end;

endfunction
