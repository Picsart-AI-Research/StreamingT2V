from PIL import Image
import torchvision.transforms as transforms
pil_to_torch = transforms.Compose([
    transforms.PILToTensor()
])


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def resize_to_fit(image, size):
    W, H = size
    w, h = image.size
    if H / h > W / w:
        H_ = int(h * W / w)
        W_ = W
    else:
        W_ = int(w * H / h)
        H_ = H
    return image.resize((W_, H_))


def pad_to_fit(image, size):
    W, H = size
    w, h = image.size
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    return add_margin(image, pad_h, pad_w, pad_h, pad_w, (0, 0, 0))


def resize_and_keep(pil_img):
    myheight = 576
    hpercent = (myheight/float(pil_img.size[1]))
    wsize = int((float(pil_img.size[0])*float(hpercent)))
    pil_img = pil_img.resize((wsize, myheight))
    return pil_img


def center_crop(pil_img):
    width, height = pil_img.size
    new_width = 576
    new_height = 576

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    pil_img = pil_img.crop((left, top, right, bottom))
    return pil_img
