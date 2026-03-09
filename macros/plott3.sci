function plott3(T,X,S,scomp)
//***************************************************
// PlotT3:
//   Plots a 2D color contour plot for the stress or
//   strain S of 3 node triangle elements, held in the 
//   topology matrix T, using the coordinate matrix X.
// Syntax:
//   plott3(T,X,S,comp)
// Input:
//   T         :  element topology matrix.
//   X         :  node coordinate matrix. (2D or 3D)
//   S         :  global stress/strain matrix.
//   scomp     :  stress/strain component to plot
// Date:
//   Version 1.0    04.05.95
//***************************************************

// number of elements in T
nel = rows(T);
// number of available stress/strain components
ncomp = cols(S);

if scomp > ncomp
  printf('\n\nRequested output component No. //d does not exist\n',scomp);
  printf('Press any key to break plotting routine.\n\n');
  halt;
  break;
end;

// initial graphics commands
ncolors=128;  // Number of colors
Smax= max(max( S(:,scomp) ));
Smin= min(min( S(:,scomp) ));

xset("colormap",jetcolormap( ncolors ))
colorbar(Smin,Smax);
ci=(Smax-Smin)/(ncolors-1);

select scomp
case 1 then
  s_dir="x";
case 2 then
  s_dir="y";
case 3 then
  s_dir="Theta";
end

xtitle("(Stress/Strain) Output in the " + s_dir + " direction")

// plot contours for all elements

Triang = [1 1 2 3 1];

// plot contours for all elements
for i=1:nel
  x = X(T(i,1:3),:);
  s = zeros(3,1)+S(i,scomp);

  smin= min(min( s ));
  smax= max(max( s ));

  ind1= ceil( (smin-Smin+ci/2)/ci );
  ind2= ceil( (smax-Smin+ci/2)/ci );
  fec(x(:,1),x(:,2),Triang,s,mesh=%t,colminmax=[ind1 ind2])
end
set(gca(),"isoview","on");
//set(gca(),"axes_visible","off");

set(gca(),"auto_clear","on");

endfunction