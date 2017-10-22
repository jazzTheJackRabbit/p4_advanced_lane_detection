from matplotlib import pyplot as plt


def rshow(image):
    plt.imshow(image)
    plt.show()


def rsave(fname, image, cmap=None):
    if cmap:
        plt.imsave("./output_images/{}.png".format(fname), image, cmap=cmap)
    else:
        plt.imsave("./output_images/{}.png".format(fname), image)

    print("Saved {} in {}.".format(fname, cmap))
