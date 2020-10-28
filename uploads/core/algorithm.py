from django.conf import settings

from cv2 import cv2
import numpy as np
import time
import sys
import os
from uploads.core.functions import add_filter, load_labeling, find_repeating_unit, resize_unit, map_skirt, refine_skirt


def preprocess():
    unit_original = cv2.imread(os.path.join(settings.MEDIA_ROOT, 'data/gebing.jpg'))
    skirt = cv2.imread(os.path.join(settings.MEDIA_ROOT, 'data/skirt.jpg'))
    l_unit_path = os.path.join(settings.MEDIA_ROOT, 'load/l_unit.txt')
    l_skirt_path = os.path.join(settings.MEDIA_ROOT, 'load/l_skirt.txt')
    shadow_path = os.path.join(settings.MEDIA_ROOT, 'data/shadow.txt')
    labels_unit, skirt_to_unit, shadows = load_labeling(l_unit_path, l_skirt_path, shadow_path)
    tone = (-80, -80, -80) # FIXME
    return unit_original, labels_unit, skirt_to_unit, shadows, skirt, tone


def generate_new_skirt(unit_original, gebing_new, labels_unit, skirt_to_unit, shadows, skirt, tone, gebing_is_unit=False):
    start = time.time()
    unit_new = gebing_new if gebing_is_unit else find_repeating_unit(gebing_new)
    print('[INFO] finding repeating unit finished after %f seconds' % (time.time()-start))
    unit_new = resize_unit(unit_new, unit_original)
    print('[INFO] resizing unit finished after %f seconds' % (time.time()-start))
    skirt_raw = map_skirt(unit_new, labels_unit, skirt_to_unit, skirt, tone)
    print('[INFO] mapping onto skirt finished after %f seconds' % (time.time()-start))
    # skirt_refined = refine_skirt(skirt_raw, shadows)
    skirt_refined = skirt_raw
    print('[INFO] shade refinement finished after %f seconds' % (time.time()-start))
    return skirt_refined


"""
    The primary method for server to run.
"""
def run():
    gebings = [os.path.join(settings.MEDIA_ROOT, 'data/gebing.jpg')]
    unit_original, labels_unit, skirt_to_unit, shadow, skirt, tone = preprocess()
    gebing_is_unit = True
    print('[INFO] data preprocessing finished')
    for g in gebings:
        gebing_new = cv2.imread(g)
        if gebing_new is None:
            raise FileNotFoundError('File %s does not exist.' % g)
        skirt_refined = generate_new_skirt(unit_original, gebing_new, labels_unit, skirt_to_unit, shadow, skirt, tone, gebing_is_unit)
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'result/%s_replaced.jpg' % g[:-4]), skirt_refined)