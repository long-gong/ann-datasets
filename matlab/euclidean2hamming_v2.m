% dataset_name should be one of: sift_1M / sift_1B / gist_1M / gist_80M.
% You should download the datasets separately.
function euclidean2hamming_v2(dataset_name, nb, word_size)

if nargin == 0
    error(['Too few arguments! Please provide at least the name' ...
        'for the dataset, which should be one of the following' ...
        ': sift_1M | sift_1B | gist_1M | gist_80M']);
end
% dataset_name = 'sift_1B';
% dataset_name = 'gist_80M';

% nb controls the lenght of the binary codes generated.
% nb can be set from outside.
if (~exist('nb', 'var'))
    nb = 64;
    fprintf(stderr, 'Using default lenght of the binary codes generated: %d\n', nb);
end

if (~exist('word_size', 'var'))
    word_size = 64;
    fprintf(stderr, 'Using default word size: %d\n', word_size);
end

switch word_size
    case 64
        data_type = 'uint64';
    case 32
        data_type = 'uint32';
    case 16
        data_type = 'uint16';
    case 8
        data_type = 'uint8';
    otherwise
        error('Unsupported word size: %d\n', word_size);
end

if mod(nb, word_size) ~= 0
    fprintf(stderr, 'Unspported number of bits (Currently we only support the cases where nb is divided by word_size): %d\n', nb);
    exit(1);
end
% Where the corresponding datasets are stored:
TINY_HOME = '../Euclidean/tiny'; 		% the root of 80 million tiny images dataset
INRIA_HOME = '../Euclidean/inria';		% the root of INIRA BIGANN datasets
% Where the output matrix of binary codes should be stored:
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
    addpath([IO_MATLAB_DIR, '/bigann']);
else
    addpath([IO_MATLAB_DIR, '/tiny_code']);
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
        trdata = fvecs_read([datahome, '/ANN_SIFT1M/sift_learn.fvecs']);
        learn_mean = mean(trdata, 2);
        save([CACHE_DIR, '/sift_1M_mean'], 'learn_mean');
    elseif strcmp(dataset_name, 'sift_1B')
        Ntraining = 10^8;
        nbuffer = 10^6;
        for i=1:floor(Ntraining/nbuffer)
            fprintf('%d/%d\r', i, floor(Ntraining/nbuffer));
            trdatai = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
            learn_meani(:,i) = sum(trdatai, 2, 'double');
        end
        learn_mean = sum(learn_meani, 2, 'double');
        learn_mean = learn_mean / Ntraining;
        clear trdatai learn_meani;
        save([CACHE_DIR, '/sift_1B_mean'], 'learn_mean');
    elseif strcmp(dataset_name, 'gist_1M')
        trdata = fvecs_read([datahome, '/ANN_GIST1M/gist_learn.fvecs']);
        learn_mean = mean(trdata, 2);
        save([CACHE_DIR, '/gist_1M_mean'], 'learn_mean');
    elseif strcmp(dataset_name, 'gist_80M')
        Ntraining = 10^7;
        nbuffer = 10^6;
        for i=1:floor(Ntraining/nbuffer)
            fprintf('%d/%d\r', i, floor(Ntraining/nbuffer));
            trdatai = read_tiny_gist_binary(((i-1)*nbuffer+1):((i)*nbuffer));
            learn_meani(:,i) = sum(trdatai, 2, 'double');
        end
        learn_mean = sum(learn_meani, 2, 'double');
        learn_mean = learn_mean / Ntraining;
        % clear trdata;
        clear trdatai learn_meani
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
if N < 10^7
    nbuffer = 10^6;
else
    nbuffer = 10^7;% if your pc has less than 16 GB RAM, please change to 10^6
end

fprintf('Computing %d-bit binary codes...\n', nb);
hdf5_fname = [outputdir, '/Hamming_mih_', num2str(nb), '_', dataset_name, '.hdf5'];
hdf5_fname_uncompact = [outputdir, '/Hamming_mih_', num2str(nb), '_', dataset_name, '_uncompact.hdf5'];
base_name = '/train';

% h5create(hdf5_fname, base_name, [N (nb / word_size)], 'Datatype', data_type);
% h5create(hdf5_fname_uncompact, base_name, [N nb], 'Datatype', 'uint8');

h5create(hdf5_fname, base_name, [(nb / word_size) N], 'Datatype', data_type);
h5create(hdf5_fname_uncompact, base_name, [nb N], 'Datatype', 'uint8');

for i=1:floor(N/nbuffer)
    fprintf('%d/%d\r', i, floor(N/nbuffer));
    if strcmp(dataset_name, 'sift_1M')
        base = fvecs_read([datahome, '/ANN_SIFT1M/sift_base.fvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'sift_1B')
        base = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_base.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'sift_1B_tr')
        base = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_learn.bvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'gist_1M')
        base = fvecs_read([datahome, '/ANN_GIST1M/gist_base.fvecs'], [(i-1)*nbuffer+1 (i)*nbuffer]);
    elseif strcmp(dataset_name, 'gist_80M')
        base = read_tiny_gist_binary( perm(((i-1)*nbuffer+1):((i)*nbuffer)) );
    end
    base = double(base);
    base = bsxfun(@minus, base, learn_mean);
    
    B1 = (W * [base; ones(1, size(base,2))]) > 0;
    h5write(hdf5_fname_uncompact, base_name, uint8(B1), [1 (i-1)*nbuffer+1], [nb nbuffer]);
    B1 = compactbit(B1, word_size);
    h5write(hdf5_fname, base_name, B1, [1 (i-1)*nbuffer+1], [(nb / word_size) nbuffer]);
end

query = [];
if strcmp(dataset_name, 'sift_1M')
    query = fvecs_read([datahome, '/ANN_SIFT1M/sift_query.fvecs']);
elseif strcmp(dataset_name, 'sift_1B')
    query = b2fvecs_read([datahome, '/ANN_SIFT1B/bigann_query.bvecs']);
elseif strcmp(dataset_name, 'gist_1M')
    query = fvecs_read([datahome, '/ANN_GIST1M/gist_query.fvecs']);
elseif strcmp(dataset_name, 'gist_80M')
    query = read_tiny_gist_binary( perm([(79302017-10000+1):79302017]) );
end
if (isempty(query))
    Q = [];
else
    query = bsxfun(@minus, query, learn_mean);
    Q = (W * [query; ones(1, size(query,2))] > 0);
    h5write_wrapper(hdf5_fname_uncompact, '/test', uint8(Q), 'Datatype', 'uint8');
    Q = compactbit(Q, word_size);
    h5write_wrapper(hdf5_fname, '/test', Q, 'Datatype', data_type);
end

clear B Q W;
fprintf('done.\n');
end
