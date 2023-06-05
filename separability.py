import numpy as np
import cv2

def separability_2ddft(inp_matrix):
    m,n = inp_matrix.shape

    #Define a zero padded array to map the FFT coefficients for both horizontal - H and vertical - V sections of the image
    
    H = np.zeros((m,n),dtype = complex)
    V = np.zeros((m,n),dtype = complex)

    #Split input image array into horizontal components to perform FFT along the rows
    h = np.vsplit(inp_matrix,m)

    for u in range(len(h)):
        f = np.fft.fft(h[u]) 
        H[u] = np.concatenate([f],axis = 0)

    #Split H array into vertical sections
    v = np.hsplit(H,n)

    for vv in range(len(v)):
        # Used np.transpose to perform FFT since an issue was arising of 
        t = np.transpose(v[vv])
        f1 = np.fft.fft(t)
        V[vv] = np.concatenate([f1],axis = 0) # Concatenates arrays in the horizontal direction
    
    # Applied the transpose to show FFT of the coefficients along the vertical direction so that the final a
    V = np.transpose(V)    
    return V



# np_fft = np.fft.fft2(img_gray)
# print(np_fft[1,0])
