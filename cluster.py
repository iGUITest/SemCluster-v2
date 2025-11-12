import random
import sys

import metrics


# View progress bar
def view_bar(num, total):
    """
    Display a progress bar in the console
    :param num: Current progress count
    :param total: Total count to complete
    """
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r[%s%s] complete: %d%%, cnt: %d' % ("=" * rate_num, " " * (100 - rate_num), rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush()


class SemiCluster:
    """
    display class
    """
    def __init__(self, cluster_k, iteration, distance_matrix, p_matrix, r_matrix, st_matrix, ct_matrix):
        self.cluster_k = cluster_k
        self.iteration = iteration
        self.distance_matrix = distance_matrix
        self.p_matrix = p_matrix
        self.r_matrix = r_matrix
        self.st_matrix = st_matrix
        self.ct_matrix = ct_matrix
        self.conflict_cnt = 0
        self.loop_cnt = 0
    # distance_matrix
    #self.distance_matrix = metrics.get_distance_matrix(self.distance_matrix)

    def getdistance(self, index1, index2):
        df = self.distance_matrix
        index_all_list = []
        for i in range(0, len(df)):
            index_all_list.append(int(df.loc[i][0]))
        num1 = index_all_list.index(index1)
        num2 = index_all_list.index(index2)
        return df.iat[num1, num2]

    # distance_matrix
    # self.distance_matrix = metrics.get_distance_matrix(self.distance_matrix)

    def rule(self, type_dict, type_k, index, must_link_dict, cannot_link_dict, euclidean_distance_list):
        if len(type_dict[type_k]) == 0 or len(type_dict[type_k]) == 1:
            return type_k
        else:
            type_k_min = type_k
            must_link_cnt_list = []
            cannot_link_cnt_list = []
            rank = [index for index, value in
                    sorted(list(enumerate(euclidean_distance_list)), key=lambda x: x[1], reverse=True)]
            for i in rank:
                must_link_cnt = 0
                cannot_link_cnt = 0
                for type_index in type_dict[i + 1]:
                    if type_index in cannot_link_dict[index]:
                        cannot_link_cnt = cannot_link_cnt + 1
                    if type_index in must_link_dict[index]:
                        must_link_cnt = must_link_cnt + 1
                must_link_cnt_list.append(must_link_cnt)
                cannot_link_cnt_list.append(cannot_link_cnt)
            for i in range(len(must_link_cnt_list)):
                if must_link_cnt_list[i] > 0 and cannot_link_cnt_list[i] == 0:
                    return rank[i] + 1
                if must_link_cnt_list[i] > 0 and cannot_link_cnt_list[i] > 0:
                    print()
                    print('Conflict1!')
                    self.conflict_cnt = self.conflict_cnt + 1
                    return type_k_min
            for i in range(len(cannot_link_cnt_list)):
                if cannot_link_cnt_list[i] > 0:
                    continue
                else:
                    return rank[i] + 1
        print()
        print('Conflict2!')
        self.conflict_cnt = self.conflict_cnt + 1
        return type_k
    # distance_matrix
    #self.distance_matrix = metrics.get_distance_matrix(self.distance_matrix)
    def randomGenerateInitList(self, index_all_list, must_link_dict):
        flag = False
        index_init_list = []
        while not flag:
            self.loop_cnt = self.loop_cnt + 1
            print()
            print("time：", self.loop_cnt)
            if self.loop_cnt > 10000:
                break
            flag = True
            index_init_list = random.sample(index_all_list, self.cluster_k)
            # ensure initial medoids do not include points that must stay together
            for index_1 in index_init_list:
                for index_2 in index_init_list:
                    if index_2 == index_1:
                        continue
                    if index_2 in must_link_dict[index_1]:
                        flag = False
        return index_init_list

    def getLinkDict(self, index_all_list):
        df_p = self.p_matrix
        df_r = self.r_matrix
        df_st = self.st_matrix
        df_ct = self.ct_matrix
        must_link_dict = {}
        cannot_link_dict = {}
        for index in index_all_list:
            must_link_dict[index] = []
            cannot_link_dict[index] = []
        for index_1 in index_all_list:
            for index_2 in index_all_list:
                if index_2 == index_1:
                    continue
                if df_st[df_st['index'] == index_1].iloc[0][str(index_2)] < 0.1 and df_p[df_p['index'] == index_1].iloc[0][str(index_2)] < 0.3:
                    must_link_dict[index_1].append(index_2)
                if df_ct[df_ct['index'] == index_1].iloc[0][str(index_2)] > 0.7 and df_r[df_r['index'] == index_1].iloc[0][str(index_2)] > 0.8:
                    cannot_link_dict[index_1].append(index_2)
        return must_link_dict, cannot_link_dict

    # unsupervised cluster
    def kmedoid_cluster(self):
        df = self.distance_matrix
        index_all_list = []
        for i in range(0, len(df)):
            index_all_list.append(int(df.loc[i][0]))
        type_dict = {}
        for i in range(1, self.cluster_k + 1):
            type_dict[i] = []
        base_cluster_center_list = random.sample(index_all_list, self.cluster_k)
        cnt = 0
        # K-Medoid
        while cnt < self.iteration:
            view_bar(cnt + 1, self.iteration)
            type_dict = {}
            for i in range(1, self.cluster_k + 1):
                type_dict[i] = []
            for index in index_all_list:
                distance_list = []
                for base_cluster_center in base_cluster_center_list:
                    distance = self.getdistance(index, base_cluster_center)
                    distance_list.append(distance)
                min_distance = min(distance_list)
                type_k = distance_list.index(min_distance) + 1
                type_dict[type_k].append(index)
            for i in range(1, self.cluster_k + 1):
                distances = []
                for point_1 in type_dict[i]:
                    if (len(type_dict[i])):
                        distances.append(self.getdistance(point_1, base_cluster_center_list[i - 1]))
                    else:
                        distances.append(0)
                now_distances = sum(distances)
                for point in type_dict[i]:
                    distances = [self.getdistance(point_1, point) for point_1 in type_dict[i]]
                    new_distances = sum(distances)
                    if new_distances < now_distances:
                        now_distances = new_distances
                        base_cluster_center_list[i - 1] = point
            cnt = cnt + 1
        return type_dict

    # semi-supervised cluster
    def semi_cluster(self):
        df = self.distance_matrix
        index_all_list = []
        for i in range(0, len(df)):
            index_all_list.append(int(df.loc[i][0]))
        must_link_dict, cannot_link_dict = self.getLinkDict(index_all_list)
        type_dict = {}
        for i in range(1, self.cluster_k + 1):
            type_dict[i] = []
        base_cluster_center_list = self.randomGenerateInitList(index_all_list, must_link_dict)
        cnt = 0
        # K-Medoid
        while cnt < self.iteration:
            view_bar(cnt + 1, self.iteration)
            type_dict = {}
            for i in range(1, self.cluster_k + 1):
                type_dict[i] = []
            for index in index_all_list:
                distance_list = []
                for base_cluster_center in base_cluster_center_list:
                    distance = self.getdistance(index, base_cluster_center)
                    distance_list.append(distance)
                min_distance = min(distance_list)
                type_k = distance_list.index(min_distance) + 1
                type_k = self.rule(type_dict, type_k, index, must_link_dict, cannot_link_dict,
                                   distance_list)
                if index in type_dict[type_k]:
                    continue
                else:
                    type_dict[type_k].append(index)
            for i in range(1, self.cluster_k + 1):
                distances = []
                for point_1 in type_dict[i]:
                    if (len(type_dict[i])):
                        distances.append(self.getdistance(point_1, base_cluster_center_list[i - 1]))
                    else:
                        distances.append(0)
                now_distances = sum(distances)
                for point in type_dict[i]:
                    distances = [self.getdistance(point_1, point) for point_1 in type_dict[i]]
                    new_distances = sum(distances)
                    if new_distances < now_distances:
                        now_distances = new_distances
                        base_cluster_center_list[i - 1] = point
            cnt = cnt + 1
        return type_dict


def semi(label, cluster_k, iteration, distance_matrix, st_matrix, ct_matrix, p_matrix, r_matrix):
    semiCluster = SemiCluster(cluster_k, iteration, distance_matrix, p_matrix, r_matrix, st_matrix, ct_matrix)
    # unsupervised cluster
    # type_dict = semiCluster.kmedoid_cluster()
    # print(type_dict)

    # semi-supervised cluster
    type_dict_semi = semiCluster.semi_cluster()
    ari, nmi, purity, RI, precision, recall = metrics.metric(label, cluster_k, type_dict_semi)
    print()
    print("ari:", ari)
    print("nmi:", nmi)
    print("purity:", purity)
    print("RI:", RI)
    print("precision:", precision)
    print("recall:", recall)
    return type_dict_semi
