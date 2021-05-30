import click
import numpy as np
import matplotlib.pyplot as plt


def get_components(X, k=None):
    eigval, C = np.linalg.eigh(X)
    eigval = abs(eigval)

    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    C = C[:, order]

    if k:
        k = int(k)
        eigval = eigval[:k]
        C = C[:, :k]

    singval = np.sqrt(eigval)
    sig = np.diag(singval)
    psig = np.diag(1 / singval)
    return C, sig, psig

def custom_svd(A, k=None):
    m = A.shape[0]
    n = A.shape[1]

    if m > n:
        corr = A.T @ A
        V, sig, psig = get_components(corr, k)
        U = A @ V @ psig

    else:
        corr = A @ A.T
        U, sig, psig = get_components(corr, k)
        V = A.T @ U @ psig

    return U, sig, V

def custom_svd_3d(A, k=None):
    recons = []
    for layer in range(A.shape[2]):
        U, sig, V = custom_svd(A[:, :, layer], k)

        recons.append(U @ sig @ V.T)

    full_recons = np.stack(recons, axis=2)
    full_recons[full_recons > 1] = 1
    full_recons[full_recons < 0] = 0

    return full_recons

def clip(U, T, V, k):
    return U[:, :k], T[:k], V[:k, :]

def library_svd(A, k=None):
    recons = []
    k = int(k) if k else min(A.shape[0], A.shape[1])
    for layer in range(A.shape[2]):
        components = np.linalg.svd(A[:, :, layer])    
        U, sig, Vh = clip(*components, k)

        recons.append(U @ np.diag(sig) @ Vh)
    
    return np.stack(recons, axis=2)

@click.command()
@click.option('-f', required=True)
@click.option('-out')
@click.option('-svd', default='custom')
@click.option('-k')
def main(f, out, svd, k):
    image = plt.imread(f)
    image = image / 255
    
    if svd == 'custom':
        if len(image.shape) == 3:
            recons = custom_svd_3d(image, k)
        else:
            U, sig, V = custom_svd(image, k)
            recons = U @ sig @ V.T
    elif svd == 'library':
        if len(image.shape) == 3:
            recons = library_svd(image, k)
        else:
            U, sig, Vh = np.linalg.svd(image)
            recons = U @ sig @ Vh
    else:
        raise ValueError

    if not out:
        plt.imshow(recons)
        plt.show()
    else:
        plt.imsave(out, recons)

if __name__ == "__main__":
    main()