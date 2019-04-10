% Example of compression of a dataset stored in the matrix X = [Nsamples, Ndimensions]
% SHparam contains the compression parameters
addpath('spectral_hashing')
% Toy dataset
Nsamples = 1000;
Ndimensions = 2;
X = rand(Nsamples, Ndimensions);

% Ttraining
SHparam.nbits = 10;
SHparam = trainSH(X, SHparam);

% Compress dataset: B = [Nsamples, ceil(Nbits/8)]
B = compressSH(X, SHparam);

% Compute distances: Dhamm = [Nsamples, Nsamples]
Dhamm = hammingDist(B, B); 