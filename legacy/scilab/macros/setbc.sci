function [K,p,ks] = setbc(K,p,C,dof)

// Ouput variables initialisation (not found in input variables)
ks=[];

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);





//***************************************************
// SetBC: 
//   Sets boundary conditions by imposing diagonal 
//   springs with ''large'' stiffness ks.
// Syntax:
//   [K,p,ks] = setbc(K,p,C) 
//   [K,p,ks] = setbc(K,p,C,dof) 
//   [K,p] = setbc(K,p,C) 
//   [K,p] = setbc(K,p,C,dof) 
// Input:
//   K   :  original global stiffness matrix.
//   p   :  global load vector.
//   C   :  constraint matrix, C = [ node1 dof1 u1 
//                                   node2 dof2 ...].
//          or for dof=1,      C = [ node1 u1
//                                   node2 ...].
//   dof :  number of dof pr. node.
// Output:
//   K   :  global stiffness matrix with springs. 
//   p   :  load vector including spring loads.
//   ks  :  stiffness of constraining springs. 
// Date:
//   Version 1.0    04.05.95
//***************************************************

// default number of dof = 1
if %nargin<4 then
  dof = 1;
end;

// Set spring stiffness.
ks = 0.1*max(diag(K));

// // Introduce constraining spring stiffness and loads.
// for i = 1:rows(C)
//   if dof==1 then
//     j = C(i,1);
//   else
//     j = (C(i,1)-1)*dof+C(i,2);
//   end;
//   K(j,j) = K(j,j) + ks;
//   p(j)   = p(j) + ks*C(i,$);
// end;

N=size(K,1);

for i = 1:size(C,1)
  if dof==1 & size(C,2)==2 then
    Cdof = C(i,1);
    cval = C(i,2);
  else
    Cdof = (C(i,1)-1)*dof+C(i,2);
    cval = C(i,3);
  end
  K(:,Cdof)=zeros(N,1);
  K(Cdof,:)=zeros(1,N);
  K(Cdof,Cdof)=ks;

  p=p-K(:,Cdof)*cval;
  p(Cdof)=ks*cval;  // the corresponding displ. will
                    // be equal to its constraint value
end

endfunction
