from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import sys
sys.path.insert(0, "../trimap_generation/")
sys.path.insert(0, "../")
from scripts.trimap_generation.saliency import get_saliency_fine_grained
import cv2 as cv
from matplotlib.pyplot import imshow
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from scripts.trimap import generate_trimap
from sklearn.cluster import DBSCAN
import warnings

T1 = 0.25


def get_median_superpixel(saliency, segments):
    n_segments = np.max(segments) + 1
    medians = []
    for segment_idx in range(n_segments):
        x, y = np.where(segments == segment_idx)
        current_superpixel_values = []
        for i, j in zip(x, y):
            current_superpixel_values.append(saliency[i, j])
        medians.append(np.median(current_superpixel_values))
    return medians


def set_value_for_superpixel(saliency, segment_indices, value):
    for x, y in segment_indices:
        saliency[x, y] = value


def color_clusters(img, segments, clusters, superpixels_ids):
    uniq = np.unique(clusters)
    for el in uniq:
        one_c = superpixels_ids[np.where(clusters == el)]
        color = list(np.random.choice(range(256), size=3))
        for s_id in one_c:
            segment_indices = zip(*np.where(segments == s_id))
            set_value_for_superpixel(img, segment_indices, color)


def just_color(img, segments, ids):
    for s_id in ids:
        segment_indices = zip(*np.where(segments == s_id))
        set_value_for_superpixel(img, segment_indices, (1, 0, 0))


def classify_superpixels_based_on_median_of_saliency(saliency, segments,
                                                     medians):
    sal = saliency.copy()
    classes = []
    for index, median in enumerate(medians):
        segment_indices = zip(*np.where(segments == index))
        classification = 0 if median < T1 else 1
        set_value_for_superpixel(sal, segment_indices, classification)
        classes.append(classification)
    return sal, np.array(classes)


def get_descriptor_for_each_superpixel(image, segments, descriptor_method,
                                       patch_size):
    n_segments = np.max(segments) + 1
    descriptors = np.empty((n_segments, 128))
    for segment_idx in range(n_segments):
        x, y = np.where(segments == segment_idx)
        x_avg = np.average(x)
        y_avg = np.average(y)

        key_point = cv.KeyPoint(x=x_avg, y=y_avg, _size=patch_size)
        _, desc = descriptor_method.compute(image,
                                            keypoints=[key_point],
                                            descriptors=None)
        descriptors[segment_idx, :] = desc
    return descriptors


def get_bhat_d_for_each_pair_superpixels(image, segments):
    n_segments = np.max(segments) + 1
    distances = np.empty((n_segments, n_segments))
    for i in range(n_segments):
        for j in range(n_segments):
            if i > j:
                continue
            d = get_bhat_distance(image, segments, i, j)
            distances[i, j] = d
            distances[j, i] = d
    return distances


def get_bhat_d_for_each_pair_superpixels_fast(hists):
    hist_r = hists[0]
    hist_g = hists[1]
    hist_b = hists[2]
    n_superpixels = len(hist_r)

    distances = np.empty((n_superpixels, n_superpixels))
    for i in range(n_superpixels):
        for j in range(n_superpixels):
            if i > j:
                continue
            hist_i = [hist_r[i], hist_g[i], hist_b[i]]
            hist_j = [hist_r[j], hist_g[j], hist_b[j]]
            dist = per_color_bhatt_d(hist_i, hist_j)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def get_indices_of_fg_bg_superpixels(classes):
    fg_indices = np.where(classes == 1)[0]
    bg_indices = np.where(classes == 0)[0]
    return fg_indices, bg_indices


def mean_distance_to_every_center(points, centers):
    distances = []
    for point in points:
        for center in centers:
            dist = np.linalg.norm(point - center)  #euclidian distance
            distances.append(dist)
    return np.median(np.array(distances))


def calc_avg_distance_to_centers(point, centers):
    distances = []
    for center in centers:
        dist = np.linalg.norm(point - center)
        distances.append(dist)
    return np.median(np.array(distances))


def clusterize_superpixels(descs, classes):
    fg, bg = get_indices_of_fg_bg_superpixels(classes)
    kmeans_fg = KMeans(n_clusters=5)
    kmeans_bg = KMeans(n_clusters=5)

    clusters_fg = kmeans_fg.fit_predict(descs[fg])
    clusters_bg = kmeans_bg.fit_predict(descs[bg])

    cluster_centers_fg = kmeans_fg.cluster_centers_
    cluster_centers_bg = kmeans_bg.cluster_centers_

    t2_fg = mean_distance_to_every_center(descs[fg], cluster_centers_bg)
    t2_bg = mean_distance_to_every_center(descs[bg], cluster_centers_fg)

    #     print(mean_distance_to_every_center(descs[fg], cluster_centers_fg))
    #     print(mean_distance_to_every_center(descs[bg], cluster_centers_bg))
    new_fg = []
    new_bg = []
    print(t2_fg, t2_bg)

    for index, point in enumerate(descs[fg]):

        avg_distance_to_bg = calc_avg_distance_to_centers(
            point, cluster_centers_bg)
        avg_distance_to_fg = calc_avg_distance_to_centers(
            point, cluster_centers_fg)
        #         print(1 - (avg_distance_to_bg  / t2_fg))
        if 1 - (avg_distance_to_bg / t2_fg) > 0.35:
            new_bg.append(fg[index])
        else:
            new_fg.append(fg[index])

    for index, point in enumerate(descs[bg]):

        avg_distance_to_fg = calc_avg_distance_to_centers(
            point, cluster_centers_fg)
        avg_distance_to_bg = calc_avg_distance_to_centers(
            point, cluster_centers_bg)

        if 1 - (avg_distance_to_fg / t2_bg) > 0.35:
            new_fg.append(bg[index])
        else:
            new_bg.append(bg[index])
    return new_fg, new_bg


def color_saliency(fg, bg, segments, saliency):
    sal = saliency.copy()
    # for index, segment_idx in enumerate(fg):
    #     segment_indices = zip(*np.where(segments == segment_idx))
    #     classification = 1
    #     set_value_for_superpixel(sal, segment_indices, classification)

    for index, segment_idx in enumerate(bg):
        segment_indices = zip(*np.where(segments == segment_idx))
        classification = 0
        set_value_for_superpixel(sal, segment_indices, classification)
    return sal


def get_saliency_refined(image, saliency_initial):
    sift = cv.xfeatures2d.SIFT_create()
    segments = slic(image, n_segments=400, compactness=20)
    medians = get_median_superpixel(1 - saliency_initial / 255, segments)
    sal, classes = classify_superpixels_based_on_median_of_saliency(
        1 - saliency_initial / 255, segments, medians)
    descs = get_descriptor_for_each_superpixel(image, segments, sift, 40)
    new_fg, new_bg = clusterize_superpixels(descs, classes)
    fg, bg = get_indices_of_fg_bg_superpixels(classes)

    return new_fg, new_bg, fg, bg


def trimap_gen(im):
    kernel_erosion = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    kernel_dilation = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilated = cv.dilate(im, kernel_dilation, iterations=10)
    eroded = cv.erode(im, kernel_erosion, iterations=10)
    unknown = dilated - eroded
    print(np.unique(eroded))
    return eroded + unknown * (128 / 255)


def per_color_bhatt_d(ps, qs):
    # ps and qs contains per color histogram
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        distances = []
        for i in range(len(ps)):
            distances.append(sum(np.sqrt(ps[i] * qs[i])))
        distance = (sum(distances) / len(distances))  
        if distance > 1:
            distance = 1

        if distance < 0:
            distance = 0
        try:
            return np.sqrt(1 - distance)
        except Warning:
            print(distance)
            return 1


def normalize_hists(hists):
    new_hists = []
    for hist in hists:
        new_hists.append(hist / (sum(hist)))
    return np.array(new_hists)


def get_bhat_distance(image, segments, segment_a, segment_b):
    hist_r = cv.calcHist(images=[image],
                         channels=[0],
                         mask=(segments == segment_a).astype(np.uint8),
                         histSize=[32],
                         ranges=[0, 256])
    hist_g = cv.calcHist(images=[image],
                         channels=[1],
                         mask=(segments == segment_a).astype(np.uint8),
                         histSize=[32],
                         ranges=[0, 256])
    hist_b = cv.calcHist(images=[image],
                         channels=[2],
                         mask=(segments == segment_a).astype(np.uint8),
                         histSize=[32],
                         ranges=[0, 256])

    hist2_r = cv.calcHist(images=[image],
                          channels=[0],
                          mask=(segments == segment_b).astype(np.uint8),
                          histSize=[32],
                          ranges=[0, 256])
    hist2_g = cv.calcHist(images=[image],
                          channels=[1],
                          mask=(segments == segment_b).astype(np.uint8),
                          histSize=[32],
                          ranges=[0, 256])
    hist2_b = cv.calcHist(images=[image],
                          channels=[2],
                          mask=(segments == segment_b).astype(np.uint8),
                          histSize=[32],
                          ranges=[0, 256])
    return per_color_bhatt_d(normalize_hists([hist_r, hist_g, hist_b]),
                             normalize_hists([hist2_r, hist2_g, hist2_b]))


def get_bhat_d_for_each_pair_superpixels(image, segments):
    n_segments = np.max(segments) + 1
    distances = np.empty((n_segments, n_segments))
    for i in range(n_segments):
        for j in range(n_segments):
            if i > j:
                continue
            d = get_bhat_distance(image, segments, i, j)
            distances[i, j] = d
            distances[j, i] = d
    return distances


def get_color_hist_for_each_superpixel(image, segments, hist_size=32):
    n_segments = np.max(segments) + 1
    hists_r = np.empty((n_segments, hist_size))
    hists_g = np.empty((n_segments, hist_size))
    hists_b = np.empty((n_segments, hist_size))
    for i in range(n_segments):
        hist_r = cv.calcHist(images=[image],
                             channels=[0],
                             mask=(segments == i).astype(np.uint8),
                             histSize=[hist_size],
                             ranges=[0, 256])
        hist_g = cv.calcHist(images=[image],
                             channels=[1],
                             mask=(segments == i).astype(np.uint8),
                             histSize=[hist_size],
                             ranges=[0, 256])
        hist_b = cv.calcHist(images=[image],
                             channels=[2],
                             mask=(segments == i).astype(np.uint8),
                             histSize=[hist_size],
                             ranges=[0, 256])
        hists_r[i] = normalize_hists([hist_r.reshape(hist_size)])
        hists_g[i] = normalize_hists([hist_g.reshape(hist_size)])
        hists_b[i] = normalize_hists([hist_b.reshape(hist_size)])
    return hists_r, hists_g, hists_b


def get_average_hists_for_cluster(cluster_indices, hists):

    hist_r, hist_g, hist_b = hists
    hist_size = hist_r.shape[1]
    avg_r, avg_b, avg_g = np.zeros(hist_size), np.zeros(hist_size), np.zeros(
        hist_size)

    for index in cluster_indices:
        current_r = hist_r[index]
        current_g = hist_g[index]
        current_b = hist_b[index]
        avg_r += current_r
        avg_g += current_g
        avg_b += current_b
    avg_r /= len(cluster_indices)
    avg_g /= len(cluster_indices)
    avg_b /= len(cluster_indices)

    return avg_r, avg_g, avg_b


def get_average_hists_for_all_clusters(segment_indices, clusters, hists):
    #no avg hist for -1 cluster
    n_clusters = np.max(clusters) + 1
    all_hists = []
    for i in range(n_clusters):
        avg_hists = get_average_hists_for_cluster(
            segment_indices[np.where(clusters == i)], hists)
        all_hists.append(avg_hists)
    return all_hists


def get_distance_one_to_all(hist, others):
    dists = []
    for hist_other in others:
        d = per_color_bhatt_d(hist, hist_other)
        dists.append(d)
    return dists


def clusterize_superpixels_hist(dists, classes, hists):
    fg, bg = get_indices_of_fg_bg_superpixels(classes)

    hist_shape = hists[0].shape
    hists_culture = np.array(list(zip(hists))).reshape(3, hist_shape[0],
                                                       hist_shape[1])
    dbscan_fg = DBSCAN(eps=0.5, metric="precomputed", min_samples=2)
    dbscan_bg = DBSCAN(eps=0.2, metric="precomputed", min_samples=2)

    clusters_fg = dbscan_fg.fit_predict(dists[fg][:, fg])
    clusters_bg = dbscan_bg.fit_predict(dists[bg][:, bg])

    avg_hists_fg = get_average_hists_for_all_clusters(fg, clusters_fg, hists)
    avg_hists_bg = get_average_hists_for_all_clusters(bg, clusters_bg, hists)

    #     t2_fg = mean_distance_to_every_center(descs[fg], cluster_centers_bg)
    #     t2_bg = mean_distance_to_every_center(descs[bg], cluster_centers_fg)

    #     print(mean_distance_to_every_center(descs[fg], cluster_centers_fg))
    #     print(mean_distance_to_every_center(descs[bg], cluster_centers_bg))
    new_fg = []
    new_bg = []
    #     print(t2_fg, t2_bg)
    new_bg += list(fg[np.where(clusters_fg == -1)])
    fg = np.delete(fg, np.where(clusters_fg == -1))
    bg = np.delete(bg, np.where(clusters_bg == -1))

    fg_hists = hists_culture[:, fg]
    bg_hists = hists_culture[:, bg]
    for index in range(fg_hists.shape[1]):
        point = fg_hists[:, index]

        dists_to_bg = get_distance_one_to_all(point, avg_hists_bg)
        dists_to_fg = get_distance_one_to_all(point, avg_hists_fg)

        if (np.min(dists_to_bg) < np.min(dists_to_fg)
            ) and np.abs(np.min(dists_to_bg) - np.min(dists_to_fg)) > 0.99:
            print(np.min(dists_to_bg), np.min(dists_to_fg), fg[index], index)
            new_bg.append(fg[index])
        else:
            new_fg.append(fg[index])

    print("BGS")
    for index in range(bg_hists.shape[1]):
        point = bg_hists[:, index]

        dists_to_bg = get_distance_one_to_all(point, avg_hists_bg)
        dists_to_fg = get_distance_one_to_all(point, avg_hists_fg)

        if (np.min(dists_to_bg) > np.min(dists_to_fg)
            ) and np.abs(np.min(dists_to_fg) - np.min(dists_to_bg)) > 0.99:
            print(np.min(dists_to_bg), np.min(dists_to_fg), bg[index], index)
            new_fg.append(bg[index])


#             print(np.min(dists_to_fg), np.min(dists_to_bg))
        else:
            new_bg.append(bg[index])
    return new_fg, new_bg