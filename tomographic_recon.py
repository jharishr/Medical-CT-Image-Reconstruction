# One object, many angles, one reconstruction
import numpy as np    
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, rotate

# synthetic phantom image (numeric 2D array)
image = np.ones([100, 100])

diag = len(np.diag(image)//2)
image = np.pad(image, pad_width=diag+10)

# simulating CT scan by adding circular features
_ = np.linspace(-1, 1, image.shape[0])
xv, yv = np.meshgrid(_,_)
image[(xv-0.1)**2 + (yv-0.2)**2<0.01] = 2

# rotating the image and summing intensities to get sinogram
img_rot = rotate(image, 45)

fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
ax[0].pcolor(xv,yv,image, shading='auto')
ax[1].pcolor(xv,yv,img_rot, shading='auto')
plt.show()

thetas = np.arange(0,180,5) * np.pi/180
rs = _
dtheta = np.diff(thetas)[0]
dr = np.diff(rs)[0]
rotations = np.array([rotate(image, theta*180/np.pi) for theta in thetas])

plt.imshow(rotations[5])

p = np.array([rotation.sum(axis=0)*dr for rotation in rotations]).T

plt.plot(rs, p[:,9])
plt.xlabel('r', fontsize=20)
plt.ylabel('$\ln(I_0/I)$', fontsize=20)
plt.show()

plt.pcolor(thetas, rs, p, shading='auto')
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel('$r$', fontsize=20)
plt.show()

# each projection Ramp Filtered and Back-projected   
p_interp = RectBivariateSpline(rs, thetas, p)

def get_fBP(x,y):
    return p_interp(x*np.cos(thetas)+y*np.sin(thetas), thetas, grid=False).sum() * dtheta

fBP = np.vectorize(get_fBP)(xv,yv)

plt.figure(figsize=(6,6))
plt.pcolor(fBP)
plt.show()

from scipy.fft import fft, ifft

P = fft(p, axis=0)
nu = np.fft.fftfreq(P.shape[0], d=np.diff(rs)[0])

P.T.shape
nu.shape

integrand = P.T * np.abs(nu)
integrand = integrand.T
p_p = np.real(ifft(integrand, axis=0))


p_p_interp = RectBivariateSpline(rs, thetas, p_p)

# output is the reconstructed 2D image array that approximates the original image
def get_f(x,y):
    return p_p_interp(x*np.cos(thetas)+y*np.sin(thetas), thetas, grid=False).sum() * dtheta

f = np.vectorize(get_f)(xv,yv)

plt.figure(figsize=(6,6))
plt.pcolor(f)
plt.show()

plt.plot(f[110])

