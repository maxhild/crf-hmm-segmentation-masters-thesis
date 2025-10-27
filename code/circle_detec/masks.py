import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplace(img):
    laplace_mask = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    img = cv2.filter2D(img, -1, laplace_mask)
    return img


def sobel(img):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    img_x = cv2.filter2D(img, -1, sobel_x)
    img_y = cv2.filter2D(img, -1, sobel_y)
    img = np.sqrt(img_x**2 + img_y**2)
    img = (img / img.max() * 255).astype(np.uint8)  # Normalize the image to 0-255
    return img


def LOG(img):
    LOG_mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])
    img = cv2.filter2D(img, -1, LOG_mask)
    return img


def gauss_kernel(size, sigma):
    if size % 2 == 0:
        size += 1
    center = size // 2
    x, y = np.meshgrid(np.arange(-center, center + 1), np.arange(-center, center + 1))
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss /= gauss.sum()
    return gauss, size

def gen_gauss_image(img, size, sigma):
    result, size = gauss_kernel(size, sigma)
    result = cv2.filter2D(img, -1, result)
    return result

def Difference_of_Gaussians(img):
    gaussian_strong = gen_gauss_image(img, 5, 1)
    gaussian_weak = gen_gauss_image(img, 5, 0.5)
    difference_of_gaussian = np.subtract(gaussian_strong, gaussian_weak)
    return difference_of_gaussian

#img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

def create_plots(img):
    Difference_of_Gaussians_img = Difference_of_Gaussians(img)
    LOG_img = LOG(img)
    sobel_img = sobel(img)
    laplace_img = laplace(img)
    
    # Create subplots and display the images
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('All My Masks')

    # Display the images
    ax1.imshow(laplace_img, cmap='gray')
    ax1.set_title('Laplace')
    ax1.axis('off')

    ax2.imshow(sobel_img, cmap='gray')
    ax2.set_title('Sobel')
    ax2.axis('off')

    ax3.imshow(LOG_img, cmap='gray')
    ax3.set_title('Laplacian of Gaussian (LOG)')
    ax3.axis('off')

    ax4.imshow(Difference_of_Gaussians_img, cmap='gray')
    ax4.set_title('Difference of Gaussians')
    ax4.axis('off')

    # Save the figure
    fig.savefig("all_my_masks.png")

    plt.show()

    '''
    Der Laplace Filter ist sehr gut darin, die Umrisse eines Objekts im Bild zu erkennen. Er erkennt jedoch kaum Kantenstärke oder Licht.
    Beim Sobel Filter werden stärkere Kanten im Bild hervorgehoben.
    Der LOG Filter ist eine Kombination aus dem Laplace Filter und dem Gauss Filter. Dadurch erkennt er ebenfalls Kantenstärken, scheint jedoch wie der Laplacefilter auch gut darin zu sein durchgehende Linien zu erkennen..
    Der Difference of Gaussians Filter ist eine Kombination aus zwei Gauss Filtern. Er ist sehr gut darin, mittelstarke Kanten im Bild zu erkennen.

    '''
    return None
