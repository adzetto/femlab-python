function K = kq4eps(K,T,X,G,S,E,mtype)
// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// KQ4EPS: 
//   Creates and assembles tangent stiffness matrix
//   for a group of elasto-plastic quadrilateral
//   4-node elements in plane stress.
// Syntax:
//   K = kq4eps(K,T,X,G,S,E)
//   K = kq4eps(K,T,X,G,S,E,type)
// Input:
//   K    : initial global stiffness matrix.
//   T    : element topology matrix.
//   X    : node coordinate matrix. 
//   G    : material property matrix. 
//   S    : current stress matrix.
//   E    : current strain matrix.
//   mtype : parameter setting material model
//          mtype = 1 -> Von Mises
//          mtype = 2 -> Drucker-Prager
// Output:
//   K    : new global tangent stiffness matrix.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// set default material model to Von Mises
if %nargin==6
  mtype = 1;
end

// if not defined - expand S 
if cols(S) ~= 16 then
  S(1,16) = 0;
end
if cols(E) ~= 16 then
  E(1,16) = 0;
end

for j = 1:rows(T)
 
  // extract element information from global arrays
  Xe = X(T(j,1:4),:);
  Ge = G(T(j,5),:); 
  // select row j and reshape into element format
  Se = matrix(S(j,:),4,4)';
  Ee = matrix(E(j,:),4,4)';

  // evaluate element stiffness 
  Ke = keq4eps(Xe,Ge,Se,Ee,mtype);

  // assemble element stiffness into global stiffness
  K  = assmk(K,Ke,T(j,:),2);
end

endfunction
