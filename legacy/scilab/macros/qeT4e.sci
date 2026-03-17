function [qe,Se,Ee] = qeT4e(Xe,Ge,Ue)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// qeT4e: 
//   Evaluates the stress and strain for the current
//   displacements. Creates the element internal
//   force vector for elastic tetrahedral 4-node
//   element.
// Syntax:
//   [qe,Se,Ee] = qeT4e(Xe,Ge,Ue)
// Input:
//   Xe   : corner coordinates (3 rows).
//   Ge   : element material data: [E , nu , (type)]
//          optional: stype = 1 : plane stress (default)
//                    stype = 2 : plane strain
//   Ue   : element displacement vector: 
//          = [u1;v1;w1; ... ;u4;v4;w4]
// Output: 
//   qe   : internal force vector.
//   Se   : element stress vector.
//   Ee   : element strain vector.
// Date:
//   Version 1.0    13.04.2007
//***************************************************

// determine number of nodes per element
nnodes = rows(Xe);

dN=[1 0 0 -1
    0 1 0 -1
    0 0 1 -1];

// Jaccobian
J = dN*Xe;
dN = J\dN;
 // B matrix is in local coordinates
    B = zeros(6,nnodes*3);

    for n=1:nnodes

      Bc = 3*(n-1)+1:3*n;

      B(:,Bc) = [dN(1,n)     0        0
                   0      dN(2,n)     0
                   0         0     dN(3,n)
                 dN(2,n)  dN(1,n)     0
                   0      dN(3,n)  dN(2,n)
                 dN(3,n)     0     dN(1,n)];
    end


// Elasticity matrix 
E  = Ge(1);
nu = Ge(2);


  D  = E/((1+nu)*(1-2*nu)) ...   
   *[ 1-nu   nu    nu     0        0           0
       nu   1-nu   nu     0        0           0
       nu    nu   1-nu    0        0           0
        0     0     0  (1-2*nu)/2  0           0
        0     0     0     0      (1-2*nu)/2    0
        0     0     0     0        0       (1-2*nu)/2];

// Form strain and stress vectors
Ee = (B*Ue)';
Se = Ee*D;

// Integrate internal force
qe = (B'*Se')*det(J);

endfunction
