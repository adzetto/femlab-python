function [] = plotq4(T,X,S,scomp)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// PlotQ4:
//   Plots a 2D color contour plot for the stress or 
//   strain S of 4 node quadrilateral elements, held 
//   in the topology matrix T, using the coordinate
//   matrix X.
// Syntax:
//   plotq4(T,X,S,comp)
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
ncomp = cols(S)/4;


if (scomp>ncomp) then
  printf("\n\nRequested output component No. %d does not exist\n",scomp);
  halt;
  break;
end;

// Inverse Gauss abscissae and weights.
r = [-1,1]*sqrt(3);
w = [1,1];
// Shape functions
for i = 1:2
  for j = 1:2
    // organize the Gauss points as the element nodes
    gp = i + 3*(j-1) - 2*(i-1)*(j-1);
    N(gp,:)  = [ (1-r(i))*(1-r(j)) (1+r(i))*(1-r(j)) ...
                 (1+r(i))*(1+r(j)) (1-r(i))*(1+r(j)) ]/4;
  end;
end;

// Extrapolation from Gauss values to nodes
Snodes=zeros(nel,4);

for i = 1:nel
  s = S(i,((1:4)-1)*ncomp + scomp);
  for j = 1:4
    Snodes(i,j) = N(j,:)*s';
  end;
end;


// initial graphics commands
// !! L. 
// colormap(mtlb(jet));
ncolors=128;  // Number of colors
Smax= max(max(Snodes));
Smin= min(min(Snodes));
try
  xset("colormap",jetcolormap( ncolors ))
catch
  disp("Can not draw the colorbar... Fix me.")
end
disp("Min. Stress = "+string(Smin))
disp("Max. Stress = "+string(Smax))

//colorbar(Smin,Smax);
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

Triang = [1 1 2 3 1;
          2 3 4 1 1];

//  set(gca(),"auto_clear","off")
for i = 1:nel
  x = X(T(i,1:4),:); 
  s = Snodes(i,1:4);

  smin= min(min( s )) ;
  smax= max(max( s )) ;

  ind1= ceil( (smin-Smin+ci/2)/ci );
  ind2= ceil( (smax-Smin+ci/2)/ci );

  fec(x(:,1),x(:,2),Triang,s,mesh=%f,colminmax=[ind1 ind2])

end
colorbar(min(Snodes),max(Snodes),[1 128])
set(gca(),"isoview","on");
//set(gca(),"axes_visible","off");

set(gca(),"auto_clear","on");

endfunction
