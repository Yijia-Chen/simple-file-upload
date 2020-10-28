from cv2 import cv2
import numpy as np
import math

################## MAIN FUNCTIONS ######################
# FIXME: works, but not yet used
def add_filter(img, tone):

    def pixelwise_operation(img, f, param):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = f(img[i][j], param)
        return img
    
    color_sin = lambda x: max(0, min(255, 255 * math.sin(math.pi * x / 510) ** 2))
    change_color = lambda color, delta: color + (color_sin(color+delta) - color_sin(color))
    change_pixel = lambda pixel, deltas: tuple(change_color(pixel[i], deltas[i]) for i in range(3))

    return pixelwise_operation(img, change_pixel, tone)

def load_labeling(lu, uts, sdw):
    units_label = []
    skirt_to_unit, shadows = {}, {}
    with open(lu, 'r') as f:
        while f:
            l = f.readline().split(' ')
            if len(l) < 8:
                break
            units_label.append(((int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (int(l[4]), int(l[5])), (int(l[6]), int(l[7][:-1]))))
    with open(uts, 'r') as f:
        while f:
            l = f.readline().split(' ')
            if len(l) < 8:
                break
            skirt_to_unit[((int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (int(l[4]), int(l[5])), (int(l[6]), int(l[7])))] = int(l[8][:-1])
    with open(sdw, 'r') as f:
        while f:
            l = f.readline().split(' ')
            if len(l) < 8:
                break
            shadows[((int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (int(l[4]), int(l[5])), (int(l[6]), int(l[7])))] = float(l[8][:-1])
    return units_label, skirt_to_unit, shadows

def find_repeating_unit(image):
    W, H = len(image), len(image[0])
    w_min, w_max = int(W / 5), int(W / 5) * 4
    h_min, h_max = int(H / 5), int(H / 5) * 4
    min_w_idx, min_w_diff = -1, 10000
    min_h_idx, min_h_diff = -1, 10000

    # try all possible repeating lengths between specified bounds
    for i in range(w_min, w_max):
        max_diff = 0

        # check similarity for all subsequent units of length i
        for j in range(i+1, W-1, i):
            img1 = image[0:min(i, W-j-1), 0:20]
            img2 = image[j:min(j+i, W-1), 0:20]
            diff = image_similarity(img1, img2, min(i, W-j-1))
            if diff > max_diff:
                max_diff = diff
        
        if min_w_diff > max_diff and max_diff < 250: # FIXME
            min_w_idx, min_w_diff = i, max_diff

    # do the same for vertical direction
    for i in range(h_min, h_max):
        max_diff = 0
        for j in range(i+1, H-1, i):
            img1 = image[0:20, 0:min(i, H-j-1)]
            img2 = image[0:20, j:min(j+i, H-1)]
            diff = image_similarity(cv2.transpose(img1), cv2.transpose(img2), min(i, H-j-1))
            if diff > max_diff:
                max_diff = diff
        if min_h_diff > max_diff and max_diff < 250:
            min_h_idx, min_h_diff = i, max_diff

    if min_w_idx == -1 or min_h_idx == -1:
        min_len = image.shape[0]
    else:
        min_len = min(min_w_idx, min_h_idx)
    cv2.imwrite('unit.jpg', image[0:min_len, 0:min_len])
    return image[0:min_len, 0:min_len]

def resize_unit(image, template):
    return cv2.resize(image, template.shape[:2], interpolation=cv2.INTER_AREA)

def map_skirt(unit, labels_unit, skirt_to_unit, skirt, tone):
    # skirt = add_filter(skirt, tone)
    for b, i in skirt_to_unit.items():
        block = warp_block(unit, skirt, labels_unit[i-1], b, tone)
        skirt = replace_roi(skirt, block)
    return skirt

def refine_skirt(skirt_raw, shadows):
    for b, dark in shadows.items():
        grayblock = warp_shadow(skirt_raw, b)
        skirt_refined = shade_roi(skirt_raw, grayblock, dark)
    return skirt_refined
########################################################

################# HELPER FUNCTIONS #####################
def encode_key(color):
    return '_'.join([str(c) for c in color])

def decode_key(key):
    return [int(k) for k in key.split('_')]

def euclid_distance(color1, color2):
    return np.sqrt(sum(np.power((np.array(color1) - np.array(color2)), 2)))

def distance(color1, color2):
    r1, g1, b1 = np.linalg.norm(color1[:,0]), np.linalg.norm(color1[:,1]), np.linalg.norm(color1[:,2])
    r2, g2, b2 = np.linalg.norm(color2[:,0]), np.linalg.norm(color2[:,1]), np.linalg.norm(color2[:,2])
    return abs(r1-r2) + abs(g1-g2) + abs(b1-b2)

def image_similarity(image1, image2, length):
    return sum([distance(r1, r2) for r1, r2 in zip(image1, image2)]) / length

def is_similiar_color(color1, color2):
    return distance(color1, color2) <= 1e-8

# TODO: fix tone adjustment later
def warp_block(unit, skirt, vertices_unit, vertices_skirt, tone):
    unit_bound = ((0, 0), (0, unit.shape[0]), (unit.shape[1], unit.shape[0]), (unit.shape[1], 0))
    homo1, _ = cv2.findHomography(np.float32(vertices_unit), np.float32(unit_bound))
    homo2, _ = cv2.findHomography(np.float32(unit_bound), np.float32(vertices_skirt))
    block_unit = cv2.warpPerspective(unit, homo1, (unit.shape[1], unit.shape[0]))
    block_skirt = cv2.warpPerspective(block_unit, homo2, (skirt.shape[1], skirt.shape[0]))
    return block_skirt

def warp_shadow(skirt_raw, vertices_shade):
    h, w = 5000, 5000
    frame = np.zeros((h, w), np.uint8)
    rect = cv2.rectangle(frame, (0, 0), (w, h), 255, -1)
    vertices_rect = ((0, 0), (0, h), (w, h), (w, 0))
    warped_rect = warp_block(rect, skirt_raw, vertices_rect, vertices_shade, None)
    return warped_rect

def replace_roi(skirt, block):
    img1, img2 = skirt, block

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(img1, img1, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    composition = cv2.add(img1_bg, img2_fg)

    return composition

def shade_roi(skirt, grayblock, factor=0.5):
    img1, img2 = skirt, grayblock

    _, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if mask[i][j] > 0:
                for k in range(3):
                    img1[i][j][k] *= factor
    
    return img1
########################################################