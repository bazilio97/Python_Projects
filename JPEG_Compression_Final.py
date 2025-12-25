import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to split a matrix of pixels into submatrices 8 x 8 size
def matsplitter( matrix, N, M):
    submatrices = []

    for i in range(0, matrix.shape[0], N):
        for j in range(0, matrix.shape[1], M):
            res = matrix[ i:i + N, j:j + M ]
            submatrices.append(res)

    return submatrices

# Function to join matrices of 8 x 8 size into one matrix
def matjoin(blocks, original_shape, N, M):

    H, W = original_shape
    out_matrix = np.zeros((H, W), dtype=blocks[0].dtype)

    block_idx = 0
    for i in range(0, H, N):
        for j in range(0, W, M):
            out_matrix[i:i + N, j:j + M] = blocks[block_idx]
            block_idx += 1

    return out_matrix

def fC( u ):
    return 1 / np.sqrt(2) if u == 1 else 1

# Forward Discrete Cosine Transform -- next (DCT)
def fDCT( mBlock ):

    nrows, ncols = mBlock.shape
    mOutBlock = np.empty( ( nrows, ncols ) )

    for u in range(1, nrows + 1):
        for v in range(1, ncols + 1):

            s = 0
            for x in range(1, nrows + 1):
                for y in range(1, ncols + 1):
                    s += np.sqrt( 2/nrows ) * np.sqrt( 2/ncols ) * fC( u ) * fC( v ) * mBlock[ x-1, y-1 ] * \
                         np.cos( np.pi/nrows * ( x-0.5 ) * ( u-1 ) ) * np.cos( np.pi/ncols * ( y-0.5 ) * ( v-1 ) )

            mOutBlock[ u-1, v-1 ] = s

    return mOutBlock

# Inverse DCT
def iDCT(mBlock):
    N, M = mBlock.shape
    mOutBlock = np.zeros((N, M))

    for x in range(1, N + 1):
        for y in range(1, M + 1):
            s = 0.0
            for u in range(1, N + 1):
                for v in range(1, M + 1):
                    s += (
                        np.sqrt(2 / N) * np.sqrt(2 / M) *
                        fC(u) * fC(v) *
                        mBlock[u - 1, v - 1] *
                        np.cos(np.pi / N * (x - 0.5) * (u - 1)) *
                        np.cos(np.pi / M * (y - 0.5) * (v - 1))
                    )
            mOutBlock[x - 1, y - 1] = s

    return mOutBlock

# Reading the initial image to get matrix of pixels to work with
img = Image.open("/here/is/the/path/to/your/image/.jpg")

# Plotting the initial image to compare quality
plt.imshow(img, cmap="gray")
plt.show()

mPix = np.array(img)
pix_blocks = matsplitter(mPix, 8, 8)

original_shape = mPix.shape

lmDCT = []
for i in range( len( pix_blocks ) ):
    mDCT = np.round( fDCT( pix_blocks[i] ) )
    lmDCT.append( mDCT )

lmDCT = np.array( lmDCT )

#q20 = np.array([
#    [16, 11, 10, 16, 24, 40, 51, 61],
#    [12, 12, 14, 19, 26, 58, 60, 55],
#    [14, 13, 16, 24, 40, 57, 69, 56],
#    [14, 17, 22, 29, 51, 87, 80, 62],
#    [18, 22, 37, 56, 68,109,103, 77],
#    [24, 35, 55, 64, 81,104,113, 92],
#    [49, 64, 78, 87,103,121,120,101],
#    [72, 92, 95, 98,112,100,103, 99]
#])
# Standart quantization table which reflect quality 20% ( q = 20 )
# The initial image quality factor = 100
q20 = np.array([
 [ 80,  55,  50,  80, 120, 200, 255, 255],
 [ 60,  60,  70,  95, 130, 255, 255, 255],
 [ 70,  65,  80, 120, 200, 255, 255, 255],
 [ 70,  85, 110, 145, 255, 255, 255, 255],
 [ 90, 110, 185, 255, 255, 255, 255, 255],
 [120, 175, 255, 255, 255, 255, 255, 255],
 [245, 255, 255, 255, 255, 255, 255, 255],
 [255, 255, 255, 255, 255, 255, 255, 255]
])

lmDCT_quantized = []
for i in range( len( lmDCT ) ):
    mDCT_quantized = np.round( lmDCT[ i ] / q20 )
    lmDCT_quantized.append( mDCT_quantized )

### DEQUANTIZATION

lmDCT_dequantized = [lmDCT_quantized[ i ] * q20 ]
for i in range( len(lmDCT_quantized) ):
    mDCT_dequantized = lmDCT_quantized[ i ] * q20
    lmDCT_dequantized.append( mDCT_dequantized )

lmPix = []
for i in range( len( lmDCT_dequantized ) ):
    mPix = iDCT( lmDCT_dequantized[ i ] )
    lmPix.append( mPix )

lmPix = np.array( lmPix )
mPix_reconstracted = matjoin(lmPix, original_shape, 8, 8)

plt.imshow(mPix_reconstracted, cmap="gray")
plt.show()



