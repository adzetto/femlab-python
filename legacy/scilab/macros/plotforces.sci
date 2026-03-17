function plotforces(T,X,P)

  if rows(X)==3
    error("3-dimensional force plot is not implemented");
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

  // get the max. Force
  maxP=max(abs( P(:,2:3) ));
  fact=0.5;
  
  // PLot Forces
  set(gca(),"auto_clear","off")

  for i = 1:size(P,1)
     x1=X( P(i,1),1 );  x2=x1 + fact*P(i,2)/maxP*dxmax;
     y1=X( P(i,1),2 );  y2=y1 + fact*P(i,3)/maxP*dymax;
     plot2d4([x1 x2],[y1 y2])
     e=gce();
     e.children.arrow_size_factor=1;
     e.children.thickness=2;
     e.children.foreground=3;
  end

endfunction