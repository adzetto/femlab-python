function [Ke] = keT4e(Xe,Ge)
warning("Tetrahedral element needs testing.")

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// KeT4E: 
//   Creates the element stiffness matrix of elastic
//   tetrahedra 4-node element.
// Syntax:
//   [Ke] = keT4e(Xe,Ge)
// Input:
//   Xe  :  corner coordinates (4 rows).
//   Ge  :  element material data: [E , nu]
// Output:
//   Ke :  element stiffness matrix.
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


// Set isotropic elasticity matrix
E = Ge(1);
nu = Ge(2);

  D  = E/((1+nu)*(1-2*nu)) ...   
   *[ 1-nu   nu    nu     0        0           0
       nu   1-nu   nu     0        0           0
       nu    nu   1-nu    0        0           0
        0     0     0  (1-2*nu)/2  0           0
        0     0     0     0      (1-2*nu)/2    0
        0     0     0     0        0       (1-2*nu)/2];

V=1/6*det(J); // volume

// Integrate element stiffness matrix.
Ke = 2*(B'*D*B)*det(J);

endfunction
