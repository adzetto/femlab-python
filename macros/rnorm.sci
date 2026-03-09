function n = rnorm(f,C,dof)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//...................................................
// rnorm:
//   Evaluates the reduced Euclidian norm of the
//   vector, f. Constrained terms in f are identified
//   from the matrix C specifies the prescribed dof. 
// Syntax:
//   n = rnorm(f,C,dof)
// Input:
//   f   : force vector
//   C   : constraint matrix
//   dof : number of dof pr. element node
// Output:
//   n   : Reduced Euclidian norm of f
// Date:
//   Version 1.0    04.05.95
//...................................................

// set up array 'cfix' marking constrained terms in f
cfix = zeros(rows(f),1);
for i=1:rows(C)
  dof_no = dof*(C(i,1)-1) + C(i,2);
  cfix(dof_no) = 1;
end

// evaluate reduced norm
n = 0;
for i=1:rows(f)
  if cfix(i) ~= 1
    n = n + f(i)^2;
  end
end
n = sqrt(n);
endfunction