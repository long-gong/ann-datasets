function h5write_wrapper( filename, datasetname, data )
%H5WRITE_WRAPPER Summary of this function goes here
%   Detailed explanation goes here
    
h5create(filename, datasetname, size(data));
h5write(filename, datasetname, data);
end

