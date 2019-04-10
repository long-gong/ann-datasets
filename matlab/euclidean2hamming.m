% dataset_name should be one of: sift_1M / sift_1B / gist_1M / gist_80M.
% You should download the datasets separately.
dataset_name = 'sift_1B';

% nb controls the lenght of the binary codes generated.
% nb can be set from outside.
if (~exist('nb', 'var'))
    nb = 128;
end

% Where the corresponding datasets are stored:
TINY_HOME = 'data/tiny'; 		% the root of 80 million tiny images dataset
% INRIA_HOME = 'data/inria';		% the root of INIRA BIGANN datasets

INRIA_HOME = '../Euclidean/inria';		% the root of INIRA BIGANN datasets
% Where the output matrix of binary codes should be stored:
% outputdir = 'codes/lsh';
outputdir = '../Hamming';

% CACHE_DIR is used to store the data mean for the datasets.
CACHE_DIR = 'cache';
IO_MATLAB_DIR = 'io';

if (~exist(outputdir, 'file'))
    mkdir(outputdir);
end
if (~exist(CACHE_DIR, 'file'))
    mkdir(CACHE_DIR);
end

% addpath matlab;
addpath('from_mih');
addpath('io');
if (strcmp(dataset_name, 'sift_1B') || strcmp(dataset_name, 'sift_1M') || strcmp(dataset_name, 'gist_1M'))
    % addpath([INRIA_HOME, '/matlab']);
    addpath([IO_MATLAB_DIR, '/bigann']);
else
    addpath([TINY_HOME, '/code']);
end

if strcmp(dataset_name, 'sift_1M')
    dataset = 'ANN_SIFT1M';
    datahome = INRIA_HOME;
    N = 10^6;
elseif strcmp(dataset_name, 'sift_1B')
    dataset = 'ANN_SIFT1B';
    datahome = INRIA_HOME;
    N = 10^9;
elseif strcmp(dataset_name, 'sift_1B_tr')
    dataset = 'ANN_SIFT1B';
    datahome = INRIA_HOME;
    N = 10^8;
elseif strcmp(dataset_name, 'gist_1M')
    dataset = 'ANN_GIST1M';
    datahome = INRIA_HOME;
    N = 10^6;
elseif strcmp(dataset_name, 'gist_80M')
    dataset = '80M';
    datahome = TINY_HOME;
    N = 79*10^6;
end

if ~exist([CACHE_DIR, '/', dataset_name, '_mean.mat'], 'file')
    fprintf('Computing the data mean for the %s dataset... \n', dataset_name);
    if strcmp(dataset_name, 'sift_1M')
        trdata = fvecs_read([datahome, '/ANN_SIFT1M/sift/sift_learn.fvecs']);
        learn_mean = mean(trdata, 2);
        save([CACHE_DIR, '/sift_1M_mean'], 'learn_mean');
    elseif strcmp(dataset_name, 'sift_1B')
        Ntraining = 10^8;
        nbuffer = 10^6;
        for i=1:floor(Ntraining/nbuffer)
            fprintf('%d/%d\r', i, floor(Ntraining/nbuffer));
            % trdatai = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
            trdatai = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
            learn_meani(:,i) = sum(trdatai, 2, 'double');
        end
        learn_mean = sum(learn_meani, 2, 'double');
        learn_mean = learn_mean / Ntraining;
        clear trdatai learn_meani;
        save([CACHE_DIR, '/sift_1B_mean'], 'learn_mean');
    elseif strcmp(dataset_name, 'gist_1M')
        trdata = fvecs_read([datahome, '/ANN_GIST1M/gist/gist_learn.fvecs']);
        learn_mean = mean(trdata, 2);
        save([CACHE_DIR, '/gist_1M_mean'], 'learn_mean');
    elseif strcmp(dataset_name, 'gist_80M')
        trdata = read_tiny_gist_binary(1:10^7);
        learn_mean = mean(trdata, 2);
        clear trdata;
        save([CACHE_DIR, '/gist_80M_mean'], 'learn_mean');
        
        perm = randperm(79302017);
        perm(1:79*10^6) = sort(perm(1:79*10^6));
        save([CACHE_DIR, '/gist_80M_mean'], 'perm', '-append');
    else
        fprintf('dataset not supported.\n');
        % continue;
        exit(-1);
    end
    fprintf('done.     \n');
else
    load([CACHE_DIR, '/', dataset_name, '_mean']);
end

nd = size(learn_mean, 1);
W = [randn(nb, nd) zeros(nb, 1)];	% Random projection-based hashing (LSH) preserves angles.
% One can load W from outside too
nbuffer = 10^7;
%B = zeros(ceil(nb/8), N, 'uint8');

fprintf('Computing %d-bit binary codes...\n', nb);
hdf5_fname = [outputdir, '/Hamming_mih_', num2str(nb), '_', dataset_name, '.hdf5'];
block_name_prefix = '/base/BLK_';
h5write_wrapper(hdf5_fname, '/number_blocks', floor(N/nbuffer));
h5write_wrapper(hdf5_fname, '/size_block', nbuffer);
% h5write_wrapper(hdf5_fname, '/data_type', 'binary');
% h5write_wrapper(hdf5_fname, '/store_type', 'uint8');

for i=1:floor(N/nbuffer)
    fprintf('%d/%d\r', i, floor(N/nbuffer));
    if strcmp(dataset_name, 'sift_1M')
        base = fvecs_read([datahome, '/ANN_SIFT1M/sift/sift_base.fvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'sift_1B')
        base = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_base.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'sift_1B_tr')
        base = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'gist_1M')
        base = fvecs_read([datahome, '/ANN_GIST1M/gist/gist_base.fvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'gist_80M')
        base = read_tiny_gist_binary( perm(((i-1)*nbuffer+1):((i)*nbuffer)) );
    end
    base = double(base);
    base = bsxfun(@minus, base, learn_mean);
    
    B1 = (W * [base; ones(1, size(base,2))]) > 0;
    B1 = compactbit(B1);
    
    datasetname = [block_name_prefix, num2str(i)];
    h5write_wrapper(hdf5_fname, datasetname, B1'); % storing in hdf5
    %B(:, (i-1)*nbuffer+1:(i)*nbuffer) = B1;
end

query = [];
if strcmp(dataset_name, 'sift_1M')
    query = fvecs_read([datahome, '/ANN_SIFT1M/sift/sift_query.fvecs']);
elseif strcmp(dataset_name, 'sift_1B')
    query = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_query.bvecs']);
elseif strcmp(dataset_name, 'gist_1M')
    query = fvecs_read([datahome, '/ANN_GIST1M/gist/gist_query.fvecs']);
elseif strcmp(dataset_name, 'gist_80M')
    query = read_tiny_gist_binary( perm([(79302017-10000+1):79302017]) );
end
if (isempty(query))
    Q = [];
else
    query = bsxfun(@minus, query, learn_mean);
    Q = (W * [query; ones(1, size(query,2))] > 0);
    Q = compactbit(Q);
end

h5write_wrapper(hdf5_fname, '/query', Q');
% fprintf('storing the codes in the file %s ...', [outputdir, '/lsh_', num2str(nb), '_', dataset_name]);
% save([outputdir, '/lsh_', num2str(nb), '_', dataset_name], 'B', 'Q', 'W', 'learn_mean', '-v7.3');
clear B Q W;

fprintf('Show me the hdf5 file info:\n');
h5disp(hdf5_fname);
fprintf('done.\n');
