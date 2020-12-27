import numpy as np

def DCT1d_Basis(k, N):
    bvec = np.cos((2*np.arange(N) + 1) * k * np.pi / (2*N))
    bvec *= np.sqrt(2/N)
    if k == 0:
        bvec /= np.sqrt(2)
    return bvec

def DCT2d_Basis(u, v, N):
    bvec_u = DCT1d_Basis(u, N)
    bvec_v = DCT1d_Basis(v, N)
    bmat = bvec_u[:, np.newaxis] @ bvec_v[np.newaxis, :]  # N x N
    #bmat *= np.sqrt(2/N)
    #bmat *= 2/N
    return bmat

# 2次元DCTの基底をならべた4次元配列を返す
#  basis[u, v] が，(u, v)の基底の値を格納した N x N array
#
def DCT2d(N):
    basis = np.empty((N, N, N, N))
    for u in range(N):
        for v in range(N):
            basis[u, v, ::] = DCT2d_Basis(u, v, N)
    return basis

