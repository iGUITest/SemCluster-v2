import image.structure_feature
import image.content_feature


def image_main(pic_dir, label_csv, xml_dir):
    st_list = image.structure_feature.getSTdis(pic_dir, label_csv)
    ct_list = image.content_feature.getCTdis(xml_dir, pic_dir, label_csv)
    return st_list, ct_list
