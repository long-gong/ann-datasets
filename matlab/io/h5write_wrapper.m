function h5write_wrapper( filename, datasetname, data, varargin )
%H5WRITE_WRAPPER Summary of this function goes here
%   Detailed explanation goes here

if ~isempty(varargin)
    h5create(filename, datasetname, size(data), varargin{:});
else
    h5create(filename, datasetname, size(data));
end
h5write(filename, datasetname, data);
end

