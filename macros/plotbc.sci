function plotbc(T,X,C)

// Number of arguments in function call
[%nargout,%nargin] = argn(0)

  if rows(X)==3
    error("3-dimensional boundary plot is not implemented");
  end

if %nargin<>3
  error("incorrect input arguments.")
end

if or( C(:,3) )
  disp("Nonzero B.C.s are drawn as red vectors.")
end

  nnodes = cols(T)-1;
  // first find the max element size
  dxmax=0; dymax=0;
  for i=1:rows(T)
     tmp=max(X( T(i,2:nnodes),1))- min( X( T(i,2:nnodes),1 ));
     if tmp>dxmax then dxmax=tmp; end
     tmp=max(X( T(i,2:nnodes),2))- min( X( T(i,2:nnodes),2 ));
     if tmp>dymax then dymax=tmp; end
  end

  fact=0.1;
  
  // PLot roller supports
  for i = 1:size(C,1)
     x=X( C(i,1),1 );
     y=X( C(i,1),2 );
     if C(i,3)==0
        if C(i,2)==1
          x=x-dxmax*fact;
        end
        if C(i,2)==2
          y=y-dymax*fact;
        end
        plot2d( x, y, -9)
     else
        if C(i,2)==1
           x=[x x+C(i,3)];
           y=[y y];
        end
        if C(i,2)==2
           x=[x x];
           y=[y y+C(i,3)];
        end
        plot2d4( x, y, 5)
        e=gce();
        e.children.arrow_size_factor=1;
        e.children.thickness=2;
     end

  end

endfunction