function [qe,Se,Ee] = qeT4e_old(Xe,Ge,Ue)

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

// Tetrahedra Volume
a = [ones(4,1) Xe];
V= 1/6*det(a);

// Formulate difference coord.
X34=X(3,:)-X(4,:);
X42=X(4,:)-X(2,:);
X23=X(2,:)-X(3,:);
X41=X(4,:)-X(1,:);
X13=X(1,:)-X(3,:);
X21=X(2,:)-X(1,:);

dN1x=  1/(6*V)*[0 X34(2) X42(2) X23(2)]*Xe(:,3);
dN1y= -1/(6*V)*[0 X34(1) X42(1) X23(1)]*Xe(:,3);
dN1z=  1/(6*V)*[0 X34(1) X42(1) X23(1)]*Xe(:,2);

dN2x=  1/(6*V)*[-X34(2) 0 -X41(2) -X13(2)]*Xe(:,3);
dN2y= -1/(6*V)*[-X34(1) 0 -X41(1) -X13(1)]*Xe(:,3);
dN2z=  1/(6*V)*[-X34(1) 0 -X41(1) -X13(1)]*Xe(:,2);

dN3x=  1/(6*V)*[-X42(2) X41(2) 0 -X21(2)]*Xe(:,3);
dN3y= -1/(6*V)*[-X42(1) X41(1) 0 -X21(1)]*Xe(:,3);
dN3z=  1/(6*V)*[-X42(1) X41(1) 0 -X21(1)]*Xe(:,2);

dN4x=  1/(6*V)*[-X23(2) X13(2) X21(2) 0]*Xe(:,3);
dN4y= -1/(6*V)*[-X23(1) X13(1) X21(1) 0]*Xe(:,3);
dN4z=  1/(6*V)*[-X23(1) X13(1) X21(1) 0]*Xe(:,2);

B=[ dN1x 0 0     dN2x 0 0     dN3x 0 0     dN4x 0 0
    0 dN1y 0     0 dN2y 0     0 dN3y 0     0 dN4y 0
    0 0 dN1z     0 0 dN2z     0 0 dN3z     0 0 dN4z
    dN1y dN1x 0  dN2y dN2x 0  dN3y dN3x 0  dN4y dN4x 0
    0 dN1z dN1y  0 dN2z dN2y  0 dN3z dN3y  0 dN4z dN4y
    dN1z 0 dN1x  dN2z 0 dN2x  dN3z 0 dN3x  dN4z 0 dN4x];

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
qe = (B'*Se')*V;

endfunction
