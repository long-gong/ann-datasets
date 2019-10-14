word_size = 64;
n_bits = [128, 256];
% dataset_names = {'sift_1B', 'gist_80M'};
dataset_names = {'sift_1M', 'gist_1M'};
for dataset_name = dataset_names
    for nb = n_bits
        fprintf('convert dataset %s to %d-bit binary codes encoding with word size %d\n', dataset_name{1}, nb, word_size);
        euclidean2hamming_v2(dataset_name{1}, nb, word_size);
    end
end