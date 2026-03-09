function varargout = devstress(varargin)
%DEVSTRESS Compatibility shim for legacy MATLAB FemLab scripts.
[varargout{1:nargout}] = devstres(varargin{:});
end
