function [qe,Se,Ee] = qeh8e(Xe,Ge,Ue)

// Ouput variables initialisation (not found in input variables)
qe=[];
Se=[];
Ee=[];

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// QeH8E: 
//   Evaluates the stress and strain for the current
//   displacements. Creates the element internal
//   force vector for elastic hexahedral 8-node
//   element.
//   If Xe contains 20 node coordinate set the inter-
//   nal force vector of 20-node element is evaluated
//   using reduced integration.
// Syntax:
//   [qe,Se,Ee] = qeh8e(Xe,Ge,Ue)
// Input:
//   Xe   : node coordinates 
//               Xe = [x1 y1; x2 y2; x3 y3; x4 y4]
//   Ge   : element material data: [E , nu , (type)]
//          optional: ptype = 1 : plane stress (default)
//                    ptype = 2 : plane strain
//   Ue   : element displacement vector: 
//               Ue = [u1;v1;u2;v2;u3;v3;u4;v4]
// Output: 
//   qe   : internal force vector.
//   Se   : element stress vector.
//   Ee   : element strain vector.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// Gauss abscissae and weights.
r = [-1,1]/sqrt(3);
w = [1,1];

// Set isotropic elasticity matrix
E = Ge(1);
nu = Ge(2);

// elastic stiffness
  D  = E/((1+nu)*(1-2*nu)) ...   
	*[ 1-nu   nu    nu     0        0           0
	    nu   1-nu   nu     0        0           0
	    nu    nu   1-nu    0        0           0
        0     0     0  (1-2*nu)/2  0           0
        0     0     0     0      (1-2*nu)/2    0
        0     0     0     0        0       (1-2*nu)/2];  

// determine number of nodes per element 
nnodes = rows(Xe);

// Initialize internal force vector
qe = zeros(3*nnodes,1);

Ee = zeros(nnodes,6);
Se = zeros(nnodes,6);

// Gauss integration of internal force vector.
gp=0;
for i = 1:2
  for j = 1:2
    for k = 1:2
      // organize the Gauss points as the element nodes
      gp = gp + 1;

    // Parametric derivatives:
    dNi =1/8*[-(1-r(j))*(1-r(k))
               (1-r(j))*(1-r(k))
               (1+r(j))*(1-r(k)) 
              -(1+r(j))*(1-r(k))
              -(1-r(j))*(1+r(k))
               (1-r(j))*(1+r(k))
               (1+r(j))*(1+r(k))
              -(1+r(j))*(1+r(k)) ];

    dNj =1/8*[-(1-r(i))*(1-r(k))
              -(1+r(i))*(1-r(k))
               (1+r(i))*(1-r(k))
               (1-r(i))*(1-r(k))
              -(1-r(i))*(1+r(k))
              -(1+r(i))*(1+r(k))
               (1+r(i))*(1+r(k))
               (1-r(i))*(1+r(k)) ];

    dNk =1/8*[-(1-r(i))*(1-r(j))
              -(1+r(i))*(1-r(j))
              -(1+r(i))*(1+r(j))
              -(1-r(i))*(1+r(j))
               (1-r(i))*(1-r(j))
               (1+r(i))*(1-r(j))
               (1+r(i))*(1+r(j))
               (1-r(i))*(1+r(j)) ];

    dN = [dNi'; dNj'; dNk'];

      if nnodes==20 then
        // evaluate the quadratic terms for the midside nodes
      error("20 node hex not supported, yet.")

        dN8 = [-r(i)*(1-r(j)),   0.5*(1-r(j)^2),  -r(i)*(1+r(j)),  -0.5*(1-r(j)^2);
              -0.5*(1-r(i)^2),   -r(j)*(1+r(i)),  0.5*(1-r(i)^2),  -r(j)*(1-r(i))  ];

        // modify corner nodes 
        dN(:,1) = dN(:,1) - 0.5*dN8(:,1) - 0.5*dN8(:,4);
        dN(:,2) = dN(:,2) - 0.5*dN8(:,2) - 0.5*dN8(:,1);
        dN(:,3) = dN(:,3) - 0.5*dN8(:,3) - 0.5*dN8(:,2);
        dN(:,4) = dN(:,4) - 0.5*dN8(:,4) - 0.5*dN8(:,3);

        // expand gradient matrix  
        dN = [dN,dN8];
      end;

      Jt = dN*Xe;
      dN = Jt\dN;

      // set up 8 node part of the gradient matrix 
      B = zeros(6,nnodes*3);

      for n=1:nnodes
      B(:,3*(n-1)+1:3*n) = [dN(1,n)     0        0
                              0      dN(2,n)     0
                              0         0     dN(3,n)
                            dN(2,n)  dN(1,n)     0
                              0      dN(3,n)  dN(2,n)
                            dN(3,n)     0     dN(1,n)];
      end

      if nnodes==20 then
      error("20 node hex not supported, yet.")
        // set up gradient matrix for midside nodes
        B8 = [dN(1,5),        0,    dN(1,6),         0,    dN(1,7),         0,    dN(1,8),         0;
                    0,  dN(2,5),          0,   dN(2,6),          0,   dN(2,7),          0,   dN(2,8);
              dN(2,5),  dN(1,5),    dN(2,6),   dN(1,6),    dN(2,7),   dN(1,7),    dN(2,8),   dN(1,8) ];

        // expand gradient matrix 
        B = [B,B8];
      end;
      // evaluate strain and stress
      Ee(gp,:) = (B*Ue)';
      Se(gp,:) = Ee(gp,:)*D;

      // evaluate internal force
      qe = qe + w(i)*w(j)*( B'*Se(gp,:)' )*det(Jt);
    end
  end;
end;

endfunction
