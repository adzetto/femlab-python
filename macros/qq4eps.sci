function [q,Sn,En] = qq4eps(q,T,X,G,u,S,E,mtype)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// QQ4EPS: 
//   Creates and assembles current internal force 
//   vector, stress matrix and strain matrix for 
//   a group of elasto-plastic quadrilateral 
//   4-node elements in plane stress.
// Syntax:
//   [q,Sn,En]  = qq4eps(q,T,X,G,u,S,E)
//   [q,Sn,En]  = qq4eps(q,T,X,G,u,S,E,type)
// Input:
//   q    : current global internal force vector.
//   T    : topology matrix for elements.
//   X    : initial node coordinate matrix. 
//   G    : element property matrix
//   u    : global displacement vector.
//   S    : current stress matrix
//   E    : current strain matrix
//   mtype : parameter setting material model
//          mtype = 1 -> Von Mises
//          mtype = 2 -> Drucker-Prager
// Output:
//   q    : internal forces from 4-node element group.
//   Sn   : updated stress matrix.
//   En   : updated strain matrix.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// set default material model to Von Mises
if %nargin==7 
  mtype = 1;
end

// if not defined - expand S and E
if cols(S) ~= 16
  S(1,16) = 0;
end
if cols(E) ~= 16
  E(1,16) = 0;
end

// reshape global displacement vector
U = reshape(u,cols(X),rows(X))';

for j = 1:rows(T)

  // define element arrays 
  Xe =  X(T(j,1:4),:);
  Ue =  U(T(j,1:4),:);
  Ue = reshape(Ue',8,1);
  Ge  = G(T(j,5),:);

  // select row j and reshape into element format
  Se = matrix(S(j,:),4,4)';
  Ee = matrix(E(j,:),4,4)';

  // evaluate internal forces for element
  [qe,Sen,Een]  = qeq4eps(Xe,Ge,Ue,Se,Ee,mtype);

  // assemble into global arrays
  q = assmq(q,qe,T(j,:),cols(X));
  Sn(j,:) = matrix(Sen',1,16); 
  En(j,:) = matrix(Een',1,16); 

end

endfunction