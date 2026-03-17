function [m] = rows(X)

// Ouput variables initialisation (not found in input variables)
m=[];

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// rows:
//   Determines number of rows in matrix X.
// Syntax:
//   m = rows(X)
// Input:
//   X :  matrix.
// Output:
//   m :  number of rows in X.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// Call MATLAB function size.
[m,n] = size(X);
endfunction
