function [Ke] = keh8e(Xe,Ge)

// Ouput variables initialisation (not found in input variables)
Ke=[];

// Display mode
mode(0);
// Display warning for floating point exception
ieee(1);

//***************************************************
// KeH8E: 
//   Creates the element stiffness matrix of elastic
//   Hexahedral 8-node element.
//   If Xe contains 20 node coordinate set the stiff-
//   ness matrix of 20-node element with quadratic 
//   shape functions is evaluated using reduced 
//   integration.
//   (8 nodes at the vertices, 12 nodes at the edges)
// Syntax:
//   Ke = keh8e(Xe,Ge)
// Input:
//   Xe   : coordinates Xe = [x1 y1; x2 y2; x3 y3; ...]
//   Ge   : element material data: [E , nu]
// Output:
//   Ke   : element stiffness matrix.
// Date:
//   Version 1.0    12.04.2007
//***************************************************  

// Gauss abscissae and weights.
r = [-1,1]/sqrt(3);
w = [1,1];

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

// determine number of nodes per element
nnodes = rows(Xe);

// Gauss integration of stiffness matrix.
for i = 1:2
 for j = 1:2
  for k = 1:2
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

       // This has to be rewritten

      error("20 node hex not supported, yet.")

      // evaluate the quadratic terms for the midside nodes
      dN20 = ... 
       [ -r(i)*(1-r(j))   0.5*(1-r(j)^2) -r(i)*(1+r(j))  -0.5*(1-r(j)^2) 
         -0.5*(1-r(i)^2) -r(j)*(1+r(i))   0.5*(1-r(i)^2) -r(j)*(1-r(i)) ];

      // modify corner nodes 
      dN(:,1) = dN(:,1) - 0.5*dN8(:,1) - 0.5*dN8(:,4);  
      dN(:,2) = dN(:,2) - 0.5*dN8(:,2) - 0.5*dN8(:,1);  
      dN(:,3) = dN(:,3) - 0.5*dN8(:,3) - 0.5*dN8(:,2);  
      dN(:,4) = dN(:,4) - 0.5*dN8(:,4) - 0.5*dN8(:,3);  

      // expand gradient matrix  
      dN = [dN, dN8];
    end;
  
    // transform to global coordinates
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
      // set up gradient matrix for midside nodes
      B20 = [dN(1,5)    0    dN(1,6)   0     dN(1,7)    0    dN(1,8)   0
                0    dN(2,5)    0    dN(2,6)    0    dN(2,7)    0    dN(2,8)
             dN(2,5) dN(1,5) dN(2,6) dN(1,6) dN(2,7) dN(1,7) dN(2,8) dN(1,8) ];

      // expand gradient matrix 
      B = [B,B20];
    end;

    Ke = Ke + w(i)*w(j)*w(k)*(B'*D*B)*det(Jt);

  end
 end;
end;


endfunction
