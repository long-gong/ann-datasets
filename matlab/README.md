# MATLAB Codes for Converting Euclidean Data to Hamming Data

Note that all the codes in this folder and its subdirectories are downloaded from from the corresponding website with slightly modification. 

## Script Codes Description

`euclidean2hamming_demo.m`: presents a demo to converting euclidean data to hamming data with `euclidean2hamming`.
`euclidean2hamming_example.m`: presents an example to converting euclidean data to hamming data with `euclidean2hamming_v2`.

## Function Codes Description

Both `euclidean2hamming` and `euclidean2hamming_v2` provide functions to converting Euclidean data to hamming (Please read the comments of them for more details). The only difference is the daat storing format of the `base` data in the output file. 

+ `euclidean2hamming`: storing the `base` data into several separate datasets with the names `base/BLK_X` where `X` is id of the block. The default block size is 10 million. If your machine does not have sufficient memory to handle that much of data, you can simply use a smaller block size. After successfully running this function, you will get one `hdf5` file with the following datasets:
    + `base/BLK_X`: base data
    + `query`: query data
    + `number_blocks`: how many blocks are there
    + `size_block`: size of the block (note that the last block might have data points less than block size)

+ `euclidean2hamming_v2`: storing the `base` data into an HUGE dataset, and using the name convention used in [ANN-Benchmarks](http://ann-benchmarks.com/). Besides it also provided a un-compact version, where each bit of the Hamming data takes 1 byte (using the format `uint8`). After successfully running this function, you will get two `hdf5` files (one for compact -- `uint64`, one for un-compact -- `uint8`) both with the following datasets:
    + `train`: base data
    + `test`: query data



## Notice

To run these codes, your MATLAB must have `hdf5` support.