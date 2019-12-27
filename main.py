import cv2
import numpy as np
import scipy.signal

def normalise(img):
    return img / img.max()

def psnr(img1, img2):
    if(img1==img2).all():
        return 100
    mse = np.mean((img1-img2)**2)
    psnr = 20*np.log10(img1.max()/np.sqrt(mse))
    return psnr

def pyrDown(img):
    kernel = cv2.getGaussianKernel(5, 1) * cv2.getGaussianKernel(5, 1).transpose()
    return scipy.signal.fftconvolve(img, kernel, 'same')[::2, ::2]

def pyrUp(img):
    kernel = cv2.getGaussianKernel(5, 1) * cv2.getGaussianKernel(5, 1).transpose()
    out = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=np.float64)
    out[::2, ::2] = img
    return 4.0 * scipy.signal.fftconvolve(out, kernel, 'same')

def getGaussian(img):
    x = np.copy(img)
    gp = [x]
    while min(x.shape) > 1:
        x = pyrDown(x)
        gp.append(x)
    return gp

def getLaplacian(img):
    gp = getGaussian(img)
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        lp.append(gp[i - 1] - pyrUp(gp[i]))
    return lp

def getImgFromLaplacian(lp):
    img = lp[0]
    for i in range(1, len(lp)):
        img = pyrUp(img)
        img = img + lp[i]
    return img

def blend(img1, img2, mask):
    lp1 = getLaplacian(img1)
    lp2 = getLaplacian(img2)
    gp = getGaussian(mask)
    gp.reverse()
    blended = []
    for l1, l2, g in zip(lp1, lp2, gp):
        rows, cols = l1.shape
        blended.append((l1 * g) + (l2 * (1- g)))
    return getImgFromLaplacian(blended)

def getHaarMatrices(size):
    Q = np.matrix('[1, 1; 1, -1]')
    M = int(size / 2)
    T = np.kron(np.eye(M), Q) / (2 ** 0.5)
    P = np.vstack((np.eye(size)[::2, :], np.eye(size)[1::2, :]))
    return P, T

def transformHaar(img):
    size = int(img.shape[0])
    trf = np.copy(img)
    for i in range(int(np.log2(size))):
        P, T = getHaarMatrices(size)
        trf[:size, :size] = P * T * trf[:size, :size] * T.transpose() * P.transpose()
        size = int(size / 2)
    return trf

def inverseTransformHaar(trf):
    img = np.copy(trf)
    size = trf.shape[0]
    start = 2
    for i in range(int(np.log2(size))):
        P, T = getHaarMatrices(start)
        img[:start, :start] = T . T * P . T * img[:start, :start] * P * T
        start = start * 2
    return img

def removeDetails(trfOrig, size, num):
    trf = np.copy(trfOrig)
    trf[size:, size:] = 0
    num -= 1
    if num == 0:
        return trf
    trf[size:, :size] = 0
    num -= 1
    if num == 0:
        return trf
    trf[:size, size:] = 0
    num -= 1
    if num == 0:
        return trf
    return removeDetails(trf, int(size / 2), num)

def addGaussianNoise(img, mean, var):
    sigma = var ** 0.5
    noisy = img + np.random.normal(mean, sigma, img.shape).reshape(img.shape)
    return noisy

def hard(trf, t):
    tmp = np.abs(trf)
    return np.where(tmp < t, 0, trf)

def soft(trf, t):
    return np.where(trf < -t, trf + t, 0) + np.where(trf > t, trf - t, 0)

if __name__ == '__main__':
    # blending two images
    # img1 = normalise(cv2.imread('images/eye.png', cv2.IMREAD_GRAYSCALE))
    # img2 = normalise(cv2.imread('images/hand.png', cv2.IMREAD_GRAYSCALE))
    # mask = normalise(cv2.imread('images/mask.png', cv2.IMREAD_GRAYSCALE))
    # blended = blend(img1, img2, mask)
    # cv2.imshow('blended.png', blended)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Haar transform
    # img = normalise(cv2.imread('images/barbara.png', cv2.IMREAD_GRAYSCALE))
    # trf = transformHaar(img)
    # img2 = inverseTransformHaar(trf)
    # cv2.imshow('haar_transformed.png', trf)
    # cv2.imshow('reconstructed', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Denoising
    # img = normalise(cv2.imread('images/barbara.png', cv2.IMREAD_GRAYSCALE))
    # noisy = addGaussianNoise(img, 0, 0.01)
    # trf = transformHaar(noisy)
    # cv2.imshow('noisy', noisy)
    # cv2.imshow('soft_denoised', inverseTransformHaar(hard(trf, 0.225)))
    # cv2.imshow('hard_denoised', inverseTransformHaar(soft(trf, 0.1)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Denoising using Laplacian
    img = normalise(cv2.imread('images/barbara.png', cv2.IMREAD_GRAYSCALE))
    noisy = addGaussianNoise(img, 0, 0.01)
    lp = getLaplacian(noisy)
    lp[0] = hard(lp[0], 0.35)
    lp[1] = hard(lp[1], 0.25)
    lp[2] = hard(lp[2], 0.225)
    lp[0] = soft(lp[0], 0.225)
    lp[1] = soft(lp[1], 0.2)
    lp[2] = soft(lp[2], 0.175)
    cv2.imshow('noisy', noisy)
    cv2.imshow('denoised', getImgFromLaplacian(lp))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Compression
    # img = normalise(cv2.imread('images/orange.png', cv2.IMREAD_GRAYSCALE))
    # toRemove = 6
    # trf = transformHaar(img)
    # comp = removeDetails(trf, int(img.shape[0] / 2), toRemove)
    # cv2.imshow('org',img)
    # cv2.imshow('haar', trf)
    # cv2.imshow('removed details', comp)

    # cv2.imshow('compressed haar', inverseTransformHaar(comp))
    # print(psnr(trf,comp))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('compressed.png', np.clip(inverseTransformHaar(comp) * 255, 0, 255))