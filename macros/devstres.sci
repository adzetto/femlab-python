function [S,Sm] = devstres(S)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

// ***************************************************
// devstress:
//   Determines the deviator and mean part of stress 
//   vector S.
// Syntax:
//   S = devstres(S)
//   [S,Sm] = devstres(S)
// Input:
//   S   : stress vector = [  S11 
//                            S22 
//                            S12 ]  
// Output:
//   S  :  deviatoric stress vector. 
//   Sm :  mean stress component.
// Date:
//   Version 1.0    04.05.95
//***************************************************

if rows(S) == 3    // plane stress
  Sm = (S(1) + S(2))/3; 
  S(1:2) = S(1:2) - Sm;
else
  Sm = (S(1) + S(2) + S(3))/3; 
  S(1:3) = S(1:3) - Sm;
end

endfunction