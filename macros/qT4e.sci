function [q,S,E] = qT4e(q,T,X,G,u)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// qT4e: 
//   Creates and assembles internal force vector,
//   stress matrix and strain matrix for a group of
//   elastic tetrahedra 4-node elements.
// Syntax:
//   [q,S,E] = qT4e(q,T,X,G,u)
// Input:
//   q  :  initial global force vector.
//   T  :  element topology matrix.
//   X  :  node coordinate matrix. 
//   G  :  material property matrix. 
//   u  :  global displacement vector.
// Output:
//   q  :  new global internal force vector.
//   S  :  global stress matrix.
//   E  :  global strain matrix.
// Date:
//   Version 1.0    13.04.2007
//***************************************************

nnodes=cols(T)-1;
dof=3;

// reshape global displacement vector
U = matrix(u,cols(X),rows(X))';

//initialize stress/strain matrices
S=zeros(rows(T),6);
E=zeros(rows(T),6);

for j = 1:rows(T)

  // define element arrays 
  Xe =  X(T(j,1:nnodes),:);
  Ue =  U(T(j,1:nnodes),:);
  Ue = matrix(Ue',dof*nnodes,1);
  Ge = G(T(j,nnodes+1),:);

  // evaluate internal forces for element
  [qe,Se,Ee]  = qeT4e(Xe,Ge,Ue);

  // assemble into global arrays
  q = assmq(q,qe,T(j,:),cols(X));

  S(j,:) = Se;
  E(j,:) = Ee; 

end

endfunction
