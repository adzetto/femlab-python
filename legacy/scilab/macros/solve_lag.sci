function [u] = solve_lag(K,p,C,dof)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// Solve_Lag: 
//   Solves the sytem matrices to obtain the Global
//   structural displacement vector by using the 
//   Lagrange Multiplier Method.
// Syntax:
//   u = solve_lag(K,p)
//   u = solve_lag(K,p,C,dof)
// Input:
//   K  :  global stiffness matrix
//   q  :  global vector.
//   C  :  constraint matrix, C = [ node1 dof1 u1 
//                                   node2 dof2 ...].
//          or for dof=1,      C = [ node1 u1
//                                   node2 ...].
//  dof :  number of dof per node.
// Output:
//   u  :  global displacement vector 
// Date:
//   Version 1.0    15.03.06
//***************************************************

if %nargin==2 then
  if type(K)==5
    // K is sparse
    u=lusolve(K,p);
  else
    u=K\p;
  end
  return;
end;

// Default number of dof = 1
if %nargin==3 then
  dof = 1;
end;

// Solve the sytem
//      Kbar * ubar = pbar
// or
//   ! K   G' !  ! u !     ! p !
//   ! G   0  !  ! L !  =  ! Q !
//
// where
//       G u = Q are the constraint equations
//       L is a vector of Lagrange Multipliers

Cr=size(C,1);
Kc=size(K,2);

ks=0.01*max(diag(K));

Q=ks*C(:,3); // the constraint value.

G=zeros( Cr, Kc );

for i = 1:Cr
  if dof==1 then
    j = C(i,1);
  else
    j = (C(i,1)-1)*dof+C(i,2);
  end;
  G(i,j)=ks;
end

Kbar = [ K  G'
         G  zeros( Cr, Cr ) ];
pbar = [ p
         Q ];

if type(K)==5
  // K is sparse
  ubar=lusolve(Kbar,pbar);
else
  ubar = Kbar\pbar;
end
u = ubar(1:Kc);

endfunction
