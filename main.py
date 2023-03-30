import pandas as pd

import cluster
import image.image_main as img
import text.text_main as txt


def avg_data(ST, CT, P, R):
    row, col = len(ST), len(ST[0])
    AVG = [ST[0]]
    for i in range(1, row):
        a_list = [ST[i][0]]
        for j in range(1, col):
            data1 = float(ST[i][j])
            data2 = float(CT[i][j])
            data3 = float(P[i][j])
            data4 = float(R[i][j])
            output_data = float(format((data1 + data2 + data3 + data4) / 4, '.4f'))
            a_list.append(output_data)
        AVG.append(a_list)
    return AVG


if __name__ == '__main__':
    # pic dir, image label file, xml dir path
    st_list, ct_list = img.image_main("file/pic_file/",
                                      "file/label_file/demo.csv",
                                      "file/xml_file/")

    # text label file path
    p_list, r_list = txt.text_main('file/label_file/demo.csv')

    # image label file path
    label = "file/label_file/evaluate.csv"

    p_arr = pd.DataFrame(p_list).values
    p_data = pd.DataFrame(p_arr[1:, 0:], columns=p_arr[0, 0:])
    r_arr = pd.DataFrame(r_list).values
    r_data = pd.DataFrame(r_arr[1:, 0:], columns=r_arr[0, 0:])
    st_arr = pd.DataFrame(st_list).values
    st_data = pd.DataFrame(st_arr[1:, 0:], columns=st_arr[0, 0:])
    ct_arr = pd.DataFrame(ct_list).values
    ct_data = pd.DataFrame(ct_arr[1:, 0:], columns=ct_arr[0, 0:])
    all_arr = pd.DataFrame(avg_data(st_list, ct_list, p_list, r_list)).values
    all_data = pd.DataFrame(all_arr[1:, 0:], columns=all_arr[0, 0:])

    type_dict = cluster.semi(label, 2, 50, all_data, st_data, ct_data, p_data, r_data)
    print("result:", type_dict)
