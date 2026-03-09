function [R] = reaction(q,C,dof)

// Ouput variables initialisation (not found in input variables)
R=[];

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);





//***************************************************
// Reaction: 
//   Extracts reaction components from the global 
//   internal force vector. 
// Syntax:
//   R = reaction(q,C,dof,comp) 
// Input:
//   q    : global internal force vector.
//   C    : constraint matrix, C = [ node1 dof1 u1 
//                                   node2 dof2 ...].
//   dof   : number of dof pr. node.
//   fcomp : force component to extract
// Output:
//   R    : reaction matrix in the format,
//                             R = [node1 dir1 R1
//                                  node2 dir2 R2 
//                                     ...   ].
//          The number of rows corresponds to 
//          the number of reactions.
// Date:
//   Version 1.0    04.05.95
//***************************************************

// Determine global dof number 
dof_no = (C(:,1)-1)*dof+C(:,2);

// Extract from internal force vector 
forces = q(dof_no);

// Organize output in a two-column matrix
R = [C(:,1:2),forces];
endfunction
