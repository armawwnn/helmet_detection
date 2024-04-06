import tensorflow as tf


def imshow(inp, title=None):
    """Imshow for Tensor. add Tensor with mean and mul with std
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)

def google_drive():
    """
    google drive 
    """
    from google.colab import drive
    drive.mount('/content/drive')

def gpu_name():
    tf.test.gpu_device_name()
