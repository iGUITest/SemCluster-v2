import os
import cv2
import json
import numpy as np
import pandas as pd
from apted import APTED, helpers
import xml.dom.minidom as minidom

import image.widget as widget


def flat_bounding(image, canny_sigma=0.33, dilate_count=4):
    """
    Find external bounding of widgets in a screenshot.

    :param image: Input image of type `ndarray`.
    :param canny_sigma: Sigma parameter for canny to control the thresholds.
    :param dilate_count: Number of iterations to perform dilation.
    :return: A list of widget bounding.
        Each bounding is represented as (x, y, w, h) where (x, y) stands for the position
        of the top-left vertex of the bounding, and (w, h) stands for the width and height of the bounding.
    """
    v = np.median(image)
    img_binary = cv2.Canny(image, int(max(0, (1 - canny_sigma) * v)), int(min(255, (1 + canny_sigma) * v)))
    img_dilated = cv2.dilate(img_binary, None, iterations=dilate_count)
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding = []
    for c in contours:
        bounding.append(cv2.boundingRect(c))
    return bounding


def process_bounding(img, bounding_list):
    non_blanks = []
    for bounding in bounding_list:
        x, y, w, h = bounding
        node = img[y:y + h, x:x + w, :]
        if not np.count_nonzero(node) == 0 and not np.count_nonzero(255 - node) == 0:
            non_blanks.append(bounding)

    enlarge_width = 5
    img_h, img_w, _ = img.shape
    enlarged_bounding = []
    for x, y, w, h in bounding_list:
        enlarged_x = max(0, x - enlarge_width)
        enlarged_y = max(0, y - enlarge_width)
        enlarged_w = min(w + 2 * enlarge_width, img_w - enlarged_x)
        enlarged_h = min(h + 2 * enlarge_width, img_h - enlarged_y)
        enlarged_bounding.append((enlarged_x, enlarged_y, enlarged_w, enlarged_h))
    return enlarged_bounding


def layout(bounding, resolution):
    """
        Data Structure Generation
    """
    # Group Generation
    groups = group_generation(basic_row_generation(bounding), resolution)

    # Row & Column Generation.

    line_merge_threshold = 1.5
    column_merge_threshold = 1.5

    for group in groups:
        group[0][0][1] = group[1]
        group[0][-1][2] = group[2]
        nodes = [(y, h) for node_row in group[0] for _, y, _, h in node_row[0]]
        lines = sorted(set([y for y, _ in nodes] + [y + h for y, h in nodes] + [group[1], group[2]]))
        merge_close_lines(lines, line_merge_threshold * resolution[1] / 100)
        lines[0] = group[1]
        lines[-1] = group[2]

        rows = []
        for top, bottom in zip(lines[:-1], lines[1:]):
            filtered_basic_rows = [row for row in group[0] if not (bottom <= row[1] or top >= row[2])]
            filtered_nodes = [(x, w) for row in filtered_basic_rows
                              for x, y, w, h in row[0] if not (y + h <= top or y >= bottom)]
            cols = sorted(set([x for x, _ in filtered_nodes] + [x + w for x, w in filtered_nodes]))
            if len(cols) == 0 or not cols[0] == 0:
                cols = [0] + cols
            if not cols[-1] == resolution[0]:
                cols.append(resolution[0])
            if len(cols) > 0:
                merge_close_lines(cols, column_merge_threshold * resolution[0] / 100)
                cols[0] = 0
                cols[-1] = resolution[0]
                cols = [[left, right] for left, right in zip(cols[:-1], cols[1:])]
            rows.append([cols, top, bottom])
        group[0] = rows

    return groups


def basic_row_generation(bounding):
    basic_rows = []
    for bounding in sorted(bounding, key=lambda b: b[1] + b[3] / 2):
        x, y, w, h = bounding
        center_y = y + h / 2
        found = False
        for row in basic_rows:
            ceiling = row[1]
            ground = row[2]
            if ceiling <= center_y <= ground:
                row[0].append(bounding)
                row[1] = min(ceiling, y)
                row[2] = max(ground, y + h)
                found = True
                break
        if not found:
            basic_rows.append([[bounding], y, y + h])
    return basic_rows


def group_generation(basic_rows, resolution):
    # Initial Group generation.
    groups = [[[row], row[1], row[2]] for row in basic_rows]
    surviving = [True] * len(groups)
    group_count = 0
    while not len(groups) == group_count:
        group_count = len(groups)
        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                if not i == j and surviving[j] and \
                        group_j[1] <= (group_i[1] + group_i[2]) / 2 <= group_j[2]:
                    group_j[0] += group_i[0]
                    group_j[1] = min(group_j[1], group_i[1])
                    group_j[2] = max(group_j[2], group_i[2])
                    surviving[i] = False
                    break
    groups = [group for i, group in enumerate(groups) if surviving[i]]

    # Group separation.
    for i, group_i in enumerate(groups):
        for group_j in groups[i + 1:]:
            if group_j[1] < group_i[2] < group_j[2]:
                group_i[2] = int((group_i[2] + group_j[1]) / 2)
                group_j[1] = group_i[2]
            elif group_j[1] < group_i[1] < group_j[2]:
                group_i[1] = int((group_i[1] + group_j[2]) / 2)
                group_j[2] = group_i[1]

    # Group simplification.
    if len(groups) > 0:
        groups[0][1] = 0
        groups[-1][2] = resolution[1]
    for prev, cur in zip(groups[:-1], groups[1:]):
        if prev[2] < cur[1]:
            cur[1] = int((prev[2] + cur[1]) / 2)
            prev[2] = cur[1]
    g_threshold = 1.5 * resolution[1] / 100
    surviving = [True] * len(groups)
    for i in range(len(groups)):
        if groups[i][2] - groups[i][1] < g_threshold:
            if i - 1 < 0 and i + 1 < len(groups):
                groups[i + 1][0] += groups[i][0]
                groups[i + 1][1] = groups[i][1]
            elif i + 1 >= len(groups) and i - 1 >= 0:
                groups[i - 1][0] += groups[i][0]
                groups[i - 1][2] = groups[i][2]
            elif i - 1 >= 0 and i + 1 < len(groups):
                height_a = groups[i - 1][2] - groups[i - 1][1]
                height_b = groups[i + 1][2] - groups[i + 1][1]
                if height_a < height_b:
                    groups[i - 1][0] += groups[i][0]
                    groups[i - 1][2] = groups[i][2]
                else:
                    groups[i + 1][0] += groups[i][0]
                    groups[i + 1][1] = groups[i][1]
            surviving[i] = False
    return [group for i, group in enumerate(groups) if surviving[i]]


def merge_close_lines(lines, threshold=5):
    i = 0
    if len(lines) < 2:
        return
    first = lines[0]
    last = lines[-1]
    while i + 1 < len(lines):
        if lines[i + 1] - lines[i] < threshold:
            lines[i] = int((lines[i] + lines[i + 1]) / 2)
            lines.pop(i + 1)
        else:
            i += 1
    if len(lines) < 2:
        lines.clear()
        lines.extend([first, last])


def exec(path):
    img = cv2.imread(path)
    groups = layout(process_bounding(img, widget.extract(path)), (img.shape[1], img.shape[0]))
    return groups


def group_json(groups):
    gson = [{
        'rows': [{
            'cols': [{
                'left': col[0],
                'right': col[1]
            } for col in row[0]],
            'top': row[1],
            'bottom': row[2]
        } for row in group[0]],
        'top': group[1],
        'bottom': group[2]
    } for group in groups]
    return json.dumps(gson, indent=2)


def GenerateXML(list_group_line, list_row_line, list_col_line, width, height, xml_path):
    impl = minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'groups', None)
    root = dom.documentElement
    len_g = len(list_group_line)
    counter = 0

    ans = ""
    list_g_str = []
    for i in range(len_g):

        str_g = ""

        group = dom.createElement("group")
        x = 0 if i == 0 else list_group_line[i - 1]
        y = 0
        w = width
        h = list_group_line[i] if i == 0 else list_group_line[i] - list_group_line[i - 1]
        group.setAttribute("x", str(x))
        group.setAttribute("y", str(y))
        group.setAttribute("w", str(w))
        group.setAttribute("h", str(h))
        root.appendChild(group)

        len_r = len(list_row_line[i]) - 1

        list_r_str = []
        for j in range(1, len_r + 1):

            row = dom.createElement("row")
            x = list_row_line[i][j - 1]
            y = 0
            w = width
            h = list_row_line[i][j] - list_row_line[i][j - 1]
            if h < 0:
                h = 0
            row.setAttribute("x", str(x))
            row.setAttribute("y", str(y))
            row.setAttribute("w", str(w))
            row.setAttribute("h", str(h))
            group.appendChild(row)

            len_c = len(list_col_line[counter])
            str_c = "{1}" * len_c
            str_r = "{1" + str_c + "}"
            list_r_str.append(str_r)
            for k in range(len_c):
                col = dom.createElement("col")
                x = list_row_line[i][j - 1]
                y = 0 if k == 0 else list_col_line[counter][k - 1]
                w = list_col_line[counter][k] if k == 0 else list_col_line[counter][k] - list_col_line[counter][k - 1]
                h = list_row_line[i][j] - list_row_line[i][j - 1]
                if h < 0:
                    h = 0
                col.setAttribute("x", str(x))
                col.setAttribute("y", str(y))
                col.setAttribute("w", str(w))
                col.setAttribute("h", str(h))
                row.appendChild(col)

            counter += 1
        for s in list_r_str:
            str_g += s
        str_g = "{1" + str_g + "}"
        list_g_str.append(str_g)

    f = open(xml_path, 'w')
    dom.writexml(f, addindent='    ', newl='\n')
    f.close()
    for s in list_g_str:
        ans += s
    ans = "{1" + ans + "}"
    return ans


def getStrs(file_dir):
    pic_files = os.listdir(file_dir)
    tree_str_list = [""] * len(pic_files)
    for pic in pic_files:
        pic_num = pic.split("_")
        num = pic_num[2].split(".")[0]

        int_num = int(num)
        img = cv2.imread(file_dir + pic)
        w, h = img.shape[1], img.shape[0]
        groups = exec(file_dir + pic)
        list_group_line = []
        list_row_line = []
        list_col_line = []
        for group in groups:
            list_group_line.append(group[2])
            cv2.line(img, (0, group[2]), (w, group[2]), (255, 0, 0), thickness=3)

        for group in groups:
            list_row_temp = []
            for i in range(len(group[0])):
                if i == 0:
                    list_row_temp.append(group[0][i][1])
                list_row_temp.append(group[0][i][2])
            list_row_line.append(list_row_temp)

        for group in groups:
            for i in range(len(group[0])):
                col_len = len(group[0][i][0])
                list_col_temp = []
                for j in range(col_len):
                    col_index = group[0][i][0][j][1]
                    list_col_temp.append(col_index)
                list_col_line.append(list_col_temp)

        tree_str_list[int_num] = GenerateXML(list_group_line, list_row_line, list_col_line, w, h,
                                             "file/xml_file/layout" + num + ".xml")

    return tree_str_list


def getSTdis(pic_dir, label_csv):
    data = pd.read_csv(label_csv, header=None).drop([0])
    title = ["index"]
    pic_list = os.listdir(pic_dir)
    for i in range(len(pic_list)):
        title.append(str(data.iloc[i][0]))
    str_list = getStrs(pic_dir)
    global_edit_distance_max = 0
    all_dist_list = [title]
    for i in range(len(str_list)):
        dist_list = [data.iloc[i][0]]
        for j in range(len(str_list)):
            src = str_list[i]
            tgt = str_list[j]
            tree1 = helpers.Tree.from_text(src)
            tree2 = helpers.Tree.from_text(tgt)
            apted = APTED(tree1, tree2)
            ted = apted.compute_edit_distance()
            if ted > global_edit_distance_max:
                global_edit_distance_max = ted
            dist_list.append(ted)
        all_dist_list.append(dist_list)

    for i in range(1, len(all_dist_list)):
        for j in range(1, len(all_dist_list[0])):
            all_dist_list[i][j] /= global_edit_distance_max
            all_dist_list[i][j] = round(all_dist_list[i][j], 4)
        all_dist_list[i][0] = int(all_dist_list[i][0])

    return all_dist_list
