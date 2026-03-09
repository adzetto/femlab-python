function [] = plotelem(T,X,str,nonum,noelem)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

// Display mode
mode(0);

// Display warning for floating point exception
ieee(1);

//***************************************************
// PlotElem:
//   Plots elements in topology matrix T with
//   coordinate matrix X. Uses linear line segment
//   between all nodes.
// Syntax:
//   plotelem(T,X)
//   plotelem(T,X[,''linetype'',nonum,noelem])
// Input:
//   T         :  element topology matrix.
//   X         :  node coordinate matrix.
//   "linetype":  string defining linetype (''y-'',''c:'',...).
//   nonum     :  if nonum = 1 -> node numbers will be plotted 
//   nnoelem   :  if noelem= 1 -> element numbers will be printed
// Date:
//   Version 1.0    04.05.95
//***************************************************

if %nargin<4 then, nonum=0; end
if %nargin<5 then, noelem=0; end


// define line style and color
if %nargin==2  then
  str1 = -4;
  str2 = 2;
else
  if ( str(1)==":" | str(1)=="-" ) then // check if line color is defined
   str1 = -4;
   str2 = str;
   str2=3;
  else
    str1 = -4;
    str2 = 4;
  end;
end;

nnodes = cols(T)-1;

// Close element contours for 3 or more nodes
//
if (cols(T)==3) then
  T(:,3) = T(:,2);
else
  T(:,cols(T)) = T(:,1);
end;

// Convert 1D problem to 2D.

if (cols(X)==1) then
  X = [X zeros(rows(X),1)];
end;

// first find the max element size
dxmax=0; dymax=0;
for i=1:rows(T)
   tmp=max(X( T(i,2:nnodes),1))- min( X( T(i,2:nnodes),1 ));
   if tmp>dxmax then dxmax=tmp; end
   tmp=max(X( T(i,2:nnodes),2))- min( X( T(i,2:nnodes),2 ));
   if tmp>dymax then dymax=tmp; end
end


// Plot 2D elements by calling function ''plot2d''
set(gca(),"auto_clear","off")

if (cols(X)==2) then
  order = [1:nnodes,1];
  if nnodes == 8
    // define node numbering for 8 node elements
    order = [1 5 2 6 3 7 4 8 1];
  end;
  // Plot nodes to scale geometry  
 // plot2d(X(:,1),X(:,2),str1)

  // Plot node numbers
  // Multiply the max value by a percentage (say 5%) to add 
  // to the node coordinate at which the number is to be written.
  frac=0.05;

  if ( nonum==1 ) then
      // draw string at lower left corner of (x,y)
      for I=1:rows(X)
        xstring(X(I,1)+dxmax*frac,X(I,2)+dymax*frac,string(I))
      end
  end

  // Plot 2D elements
  for j = 1:rows(T)
    plot2d(X(T(j,order),1),X(T(j,order),2),str2)

    // plot element number
    if (noelem==1)
        xcoord=1/nnodes*sum( X(T(j,order(1:$-1)),1) );
        ycoord=1/nnodes*sum( X(T(j,order(1:$-1)),2) );
        //xset('color',2)
        xstring(xcoord,ycoord,string(j))
        //xset('default')
    end
  end

end

// Plot 3D elements by calling function ''plot3''
if  cols(X) == 3
  // Plot nodes to scale geometry 
 // plot3d(X(:,1),X(:,2),X(:,3),str1)

  // Plot 3D elements
  select nnodes
    case 4 then   // Tetrahedra
                order = [1 2 3 1 4 2 3 4];
    case 8 then   // Hexahedra 
                order = [1 2 3 4 1 5 6 7 8 5 8 4 3 7 6 2];
    case 10 then  // Quadratic Tetrahedra
                order = [1 5 2 6 3 7 1 8 4 10 2 6 3 9 4];
  end

  for j = 1:rows(T)
    plot3d(X(T(j,order),1),X(T(j,order),2),X(T(j,order),3),flag=[0 2 4])
  end;
end;

set(gca(),"isoview","on");
// for use in ver. 4.0
//set(gca(),"axes_visible",["off","off","off"])

// enable the plot to be overwritten
set(gca(),"auto_clear","on")
endfunction
