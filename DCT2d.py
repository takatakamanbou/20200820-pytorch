import numpy as np

def DCT1d_Basis(k, N):
    bvec = np.cos((2*np.arange(N) + 1) * k * np.pi / (2*N))
    bvec *= np.sqrt(2/N)
    if k == 0:
        bvec /= np.sqrt(2)
    return bvec

'''
D = 8
b1d_0 = DCT1d_Basis(0, D)
b1d_1 = DCT1d_Basis(1, D)
b1d_2 = DCT1d_Basis(2, D)
'''

def DCT2d_Basis(u, v, N):
    bvec_u = DCT1d_Basis(u, N)
    bvec_v = DCT1d_Basis(v, N)
    bmat = bvec_u[:, np.newaxis] @ bvec_v[np.newaxis, :]  # N x N
    #bmat *= np.sqrt(2/N)
    #bmat *= 2/N
    return bmat

'''
D = 8
b2d_00 = DCT2d_Basis(0, 0, D)
b2d_01 = DCT2d_Basis(0, 1, D)
b2d_11 = DCT2d_Basis(1, 1, D)
'''

# 2次元DCTの基底をならべた4次元配列を返す
#  basis[u, v] が，(u, v)の基底の値を格納した N x N array
#
def DCT2d(N):
    basis = np.empty((N, N, N, N))
    for u in range(N):
        for v in range(N):
            basis[u, v, ::] = DCT2d_Basis(u, v, N)
    return basis

'''
basis = DCT2d(D)

W = np.empty((64, 3, 8, 8))
for i in range(8):
    for j in range(8):
        W[i*8+j, 0, ::] = basis[i, j, ::]
        W[i*8+j, 1, ::] = basis[i, j, ::]
        W[i*8+j, 2, ::] = basis[i, j, ::]
'''
