import os
import cv2
import pandas as pd
import xml.dom.minidom as xmldom

import image.vgg16 as vgg16
import image.vgg16_stub as vgg16_stub

threshold = 0.7


def cal_iou_ok(list1, list2, bias=0):
    col_min_a, row_min_a, col_max_a, row_max_a = int(list1[1]), int(list1[0]), \
                                                 int(list1[1] + list1[2]), int(list1[0] + list1[3])
    col_min_b, row_min_b, col_max_b, row_max_b = int(list2[1]), int(list2[0]), \
                                                 int(list2[1] + list2[2]), int(list2[0] + list2[3])
    if col_min_a > col_max_b or col_min_b > col_max_a or row_min_a > row_max_b or row_min_b > row_max_a:
        return False
    col_min_s = max(col_min_a - bias, col_min_b - bias)
    row_min_s = max(row_min_a - bias, row_min_b - bias)
    col_max_s = min(col_max_a + bias, col_max_b + bias)
    row_max_s = min(row_max_a + bias, row_max_b + bias)
    w = max(0, col_max_s - col_min_s)
    h = max(0, row_max_s - row_min_s)
    inter = w * h
    area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
    area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
    iou = inter / (area_a + area_b - inter)
    return iou >= threshold


def is_pure_color(com):
    baseline = com[0, 0]
    base_b = baseline[0]
    base_g = baseline[1]
    base_r = baseline[2]
    height, width = com.shape[0], com.shape[1]
    for i in range(height):
        for j in range(width):
            cur_pixel = com[i, j]
            cur_b = cur_pixel[0]
            cur_g = cur_pixel[1]
            cur_r = cur_pixel[2]
            if cur_b != base_b or cur_g != base_g or cur_r != base_r:
                return False
    return True


def process(xmlpath1, xmlpath2, imgpath1, imgpath2):
    dom_obj1 = xmldom.parse(xmlpath1)
    dom_obj2 = xmldom.parse(xmlpath2)
    element_obj1 = dom_obj1.documentElement
    element_obj2 = dom_obj2.documentElement
    sub_element_obj1 = element_obj1.getElementsByTagName("col")
    sub_element_obj2 = element_obj2.getElementsByTagName("col")
    list_file1_all = []
    list_file2_all = []

    isChange = False

    if len(sub_element_obj1) < len(sub_element_obj2):
        sub_element_obj1, sub_element_obj2 = sub_element_obj2, sub_element_obj1
        isChange = True
        for i in range(len(sub_element_obj1)):
            list_temp = [sub_element_obj1[i].getAttribute("x"),
                         sub_element_obj1[i].getAttribute("y"),
                         sub_element_obj1[i].getAttribute("w"),
                         sub_element_obj1[i].getAttribute("h")]
            list_file1_all.append(list_temp)
        for i in range(len(sub_element_obj2)):
            list_temp = [sub_element_obj2[i].getAttribute("x"),
                         sub_element_obj2[i].getAttribute("y"),
                         sub_element_obj2[i].getAttribute("w"),
                         sub_element_obj2[i].getAttribute("h")]
            list_file2_all.append(list_temp)
    else:
        for i in range(len(sub_element_obj1)):
            list_temp = [sub_element_obj1[i].getAttribute("x"),
                         sub_element_obj1[i].getAttribute("y"),
                         sub_element_obj1[i].getAttribute("w"),
                         sub_element_obj1[i].getAttribute("h")]
            list_file1_all.append(list_temp)
        for i in range(len(sub_element_obj2)):
            list_temp = [sub_element_obj2[i].getAttribute("x"),
                         sub_element_obj2[i].getAttribute("y"),
                         sub_element_obj2[i].getAttribute("w"),
                         sub_element_obj2[i].getAttribute("h")]
            list_file2_all.append(list_temp)

    count = 0
    flags = [False] * len(list_file2_all)

    match_pool = []

    for i in range(len(list_file2_all)):
        for j in range(len(list_file1_all)):
            if cal_iou_ok(list_file1_all[j], list_file2_all[i]) and flags[i] == False:
                list_t = []
                count += 1
                flags[i] = True
                list_t.append(list_file1_all[j])
                list_t.append(list_file2_all[i])
                match_pool.append(list_t)
                break

    list_match_com = []
    if isChange:
        img1 = cv2.imread(imgpath2)
        img2 = cv2.imread(imgpath1)
    else:
        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)

    reduce_count = count

    for i in range(count):
        list_temp = []
        list_pairs = match_pool[i]
        list_pair1 = list_pairs[0]
        list_pair2 = list_pairs[1]
        for k in range(4):
            list_pair1[k] = int(list_pair1[k])
            list_pair2[k] = int(list_pair2[k])
        if list_pair1[2] == 0 or list_pair1[3] == 0 or list_pair2[2] == 0 or list_pair2[3] == 0:
            continue
        com1 = img1[list_pair1[0]:list_pair1[0] + list_pair1[3],
               list_pair1[1]:list_pair1[1] + list_pair1[2]]
        if is_pure_color(com1):
            reduce_count -= 1
            continue
        com1 = cv2.resize(com1, (224, 224))
        com2 = img2[list_pair2[0]:list_pair2[0] + list_pair2[3],
               list_pair2[1]:list_pair2[1] + list_pair2[2]]
        com2 = cv2.resize(com2, (224, 224))
        list_temp.append(com1)
        list_temp.append(com2)
        list_match_com.append(list_temp)

    distance_list=vgg16.getdistance(list_match_com)

    result = 0
    for i in range(len(distance_list)):
        result += distance_list[i]
    if reduce_count == 0:
        result = 1
    if reduce_count != 0:
        result /= reduce_count
    if imgpath1 == imgpath2:
        result = 0.0
    return result


def getCTdis(xml_dir, img_dir, label_csv):
    data = pd.read_csv(label_csv, header=None).drop([0])
    title = ["index"]
    xml_list = os.listdir(xml_dir)
    xml_list.remove("empty.txt")
    img_list = os.listdir(img_dir)

    for i in range(len(xml_list)):
        title.append(str(data.iloc[i][0]))

    all_dist_list = [title]

    for i in range(len(xml_list)):
        dist_list = [data.iloc[i][0]]
        for j in range(len(xml_list)):
            xml_cur1 = xml_dir + "layout" + str(i) + ".xml"
            xml_cur2 = xml_dir + "layout" + str(j) + ".xml"
            img_cur1 = img_dir + img_list[i]
            img_cur2 = img_dir + img_list[j]
            dis_cur = process(xml_cur1, xml_cur2, img_cur1, img_cur2)
            dis_cur = round(dis_cur, 4)
            dist_list.append(dis_cur)
        all_dist_list.append(dist_list)

    return all_dist_list
