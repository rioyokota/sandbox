typedef void block_func(int, int, int *, int *, void *, void *, void *);

typedef struct
{
    char symm; // 'N' if nonsymmetric problem, 'S' if symmetric
    char dtype; // data type of problem ('s', 'd', 'c', 'z')
    int rows, cols; // number of rows and columns of a matrix
    int *row_order, *col_order; // permutation of rows and columns
    void *row_data, *col_data; // objects behind rows and columns
    int brows, bcols; // number of block rows and block columns
    int *brow_start, *bcol_start; // start point of block rows and block
        // columns (in correspondance to row_order and col_order)
    int *brow_size, *bcol_size; // size of each block row and block column
    int bcount, *bindex, *brank; // total number of blocks,
        // each block index (pair of block coordinates) and rank
    void **U, **V, **A; // buffers to save low rank approximation of a block
        // in UV format or, if block is not low rank, block itself as a
        // submatrix A. Right now it is done as a simple vector of pointers to
        // corresponding data. In future, it is planned to change it.
    block_func *kernel; // block kernel function, generating matrix
} STARS_blrmat;
// STARS_blrmat stands for STARS block low-rank matrix



STARS_blrmat *STARS_blr_compress_uniform(int bcount, int bsize, void *data,
        block_func kernel, double tol);

STARS_blrmat *STARS_blr_compress(int symm, int rows, int cols,
        void *row_data, void *col_data, int *row_order, int *col_order,
        int brows, int bcols, int *brow_start, int *bcol_start, int *brow_size,
        int *bcol_size, block_func kernel, double tol);

void STARS_blr__compress_algebraic_svd(STARS_blrmat *A, double tol);
