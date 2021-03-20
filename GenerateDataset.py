# coding=UTF-8
import json
import numpy as np
import pickle
import collections
import random
from torch.utils.data import DataLoader
from datetime import datetime

userfile = './data_file/user.json'
coursefile = './data_file/course.json'
videofile = './data_file/video.json'
schoolfile = './data_file/school.json'
teacher_file = './data_file/teacher.json'
teacher_course_file = './data_file/teacher-course.json'
school_course_file = './data_file/school-course.json'
user_course_file = './data_file/user-course.json'
course_video_file = './data_file/course-video.json'
user_video_file = './data_file/user-video.json'
concept_file = './data_file/concept.json'


class dataGenerator(object):
    def __init__(self, user_filter=15, course_filter=5):
        self.user_filter = user_filter
        self.course_filter = course_filter
        # self.getConcepts(concept_file)
        self.user_history, self.num_user, self.course_set, self.num_course, self.interactions = self.getUsers(userfile)
        self.getCourses(user_course_file)
        self.init_course_dict()

        if self.user_filter>0 and self.course_filter>0:
            self.reloadUsers()

        self.schools, self.num_school = self.getSchools(school_course_file)
        self.teachers, self.num_teacher = self.getTeachers(teacher_course_file)
        self.videos, self.num_video = self.getVideos(course_video_file)
        self.get_course_name(coursefile)

        self.getDataset()

    def getUsers(self, userfile):
        """

        :param userfile:
        get users whose interactions larger than self.user_filter
        :return: user dict:{u_id:[[course_order],[enroll_time]]}
        num_user
        course_set=[course_id1,course_id2,...]
        num_courses
        num_interactions
        """
        print('start load users')
        with open(userfile, 'rb') as file:

            # dict = collections.defaultdict(list)
            dict = {}
            course_set = []
            interactions = 0
            usercnt = 0
            for line in file.readlines():
                # if len(dict) > 10: break
                dic = json.loads(line)
                user_id = dic['id']
                course_order = dic['course_order']  # delete users whose actions less than 5
                enroll_time = dic['enroll_time']
                if len(course_order) < self.user_filter:
                    continue
                if user_id not in dict:
                    dict[user_id] = [[], [], []]
                    dict[user_id][0] = course_order
                    dict[user_id][1] = enroll_time
                    course_set += course_order
                    interactions += len(course_order)
                    usercnt += 1
                else:
                    print(dic)
                    continue

            course_set = list(set(course_set))
            coursercnt = len(course_set)
            print('load users done')
            return dict, usercnt, course_set, coursercnt, interactions

    def getCourses(self, u_cfile):
        """

        :param u_cfile:
        :return:过滤少于course_filter用户数量的课程
        """
        print('start load courses')
        course_dict = collections.defaultdict(list)
        with open(u_cfile, 'r', encoding='utf-8') as file:  # 用户
            for line in file.readlines():
                l = line.strip().split('\t')
                user, course_id = l[0], l[1]
                if user in self.user_history and course_id in self.course_set:
                    if user not in course_dict[course_id]:
                        course_dict[course_id].append(user)
        print('get course-user done')
        # print(len(course_dict))
        print('start filter courses')
        cnt_u = 0
        n_us = []
        max_u, min_u = 0, 10000
        for c in course_dict:
            l = len(course_dict[c])
            if l < self.course_filter:
                self.course_set.remove(c)
                continue
            # print(l)
            n_us.append(l)
            cnt_u += l
            max_u = max(max_u, l)
            min_u = min(min_u, l)
        mean_u = cnt_u / len(course_dict)
        self.num_course = len(self.course_set)
        print('current num course', self.num_course)
        print(mean_u, max_u, min_u)

        return n_us, mean_u, max_u, min_u

    def get_course_name(self, course_file):
        self.courseid2name = dict()
        with open(course_file, 'r', encoding='utf-8') as file:  # 添加老师
            for line in file.readlines():
                dic = json.loads(line)
                course_name = dic['name']
                course_id =dic['id']
                self.courseid2name[course_id] = course_name


    def reloadUsers(self):
        """

        :return: 根据过滤后的course，对用户进行二次筛选
        """

        new_dict = {}
        print('num user before reload', self.num_user)
        l = 0
        for u in self.user_history:
            course_seq = self.user_history[u][0].copy()
            for c in course_seq:
                if c not in self.course_set:
                    idx = self.user_history[u][0].index(c)
                    self.user_history[u][0].pop(idx)
                    self.user_history[u][1].pop(idx)
            if len(self.user_history[u][0]) >= self.user_filter:
                l += len(self.user_history[u][0])
                new_dict[u] = [[], [], []]
                new_dict[u][0] = self.user_history[u][0]
                new_dict[u][1] = self.user_history[u][1]
        self.user_history = new_dict
        self.num_user = len(self.user_history)
        self.interactions = l

        print('num user after reload', self.num_user)

    def showInfo(self):
        print('user_filter', self.user_filter, 'course_filter', self.course_filter)
        print('num_user', self.num_user)
        print('num_course', self.num_course)
        print('num_interactions', self.interactions)
        print('density', self.interactions / (self.num_course * self.num_user))
        print('num_school', self.num_school)
        print('num_teacher', self.num_teacher)
        print('num_video', self.num_video)

    def getSchools(self, s_cfile):
        """

        :param s_cfile:
        :return: 根据筛选后的课程，建立学校集合
        """
        schools = []
        with open(s_cfile, 'r', encoding='utf-8') as file:  # 添加学校
            for line in file.readlines():
                l = line.strip().split('\t')
                school, course_id = l[0], l[1]
                if course_id not in self.course_set:
                    continue
                else:
                    if school not in schools:
                        schools += [school]
                    if school not in self.course_dict[course_id][1]:
                        self.course_dict[course_id][1].append(school)
            school_cnt = len(schools)
        return schools, school_cnt

    def getVideos(self, c_vfile):
        """

        :param c_vfile:课程-视频文件
        :return: 根据筛选后的课程，建立视频集合
        """
        videos = []
        with open(c_vfile, 'r', encoding='utf-8') as file:  # 添加视频
            for line in file.readlines():
                l = line.strip().split('\t')
                course_id, video_id = l[0], l[1]
                if course_id not in self.course_set:
                    continue
                else:
                    if video_id not in videos:
                        videos += [video_id]
                    if video_id not in self.course_dict[course_id][3]:
                        self.course_dict[course_id][3].append(video_id)
            video_cnt = len(videos)

        return videos, video_cnt

    def getTeachers(self, t_cfile):
        """

        :param s_cfile:
        :return: 根据筛选后的课程，建立老师集合
        """
        teachers = []
        with open(t_cfile, 'r', encoding='utf-8') as file:  # 添加老师
            for line in file.readlines():
                l = line.strip().split('\t')
                teacher, course_id = l[0], l[1]
                if course_id not in self.course_set:
                    continue
                else:
                    if teacher not in teachers:
                        teachers += [teacher]
                    if teacher not in self.course_dict[course_id][2]:
                        self.course_dict[course_id][2].append(teacher)
            teacher_cnt = len(teachers)
        return teachers, teacher_cnt

    def getconcepts(self, con_file):
        """

        :param s_cfile:
        :return: 根据筛选后的课程，建立老师集合
        """
        concepts = []
        with open(con_file, 'r', encoding='utf-8') as file:  # 添加老师
            for line in file.readlines():
                dic = json.loads(line)
                concept_name = dic['name']
                concepts.append(concept_name)
        concept_cnt = len(concepts)
        self.concepts, self.num_concept = concepts, concept_cnt
        print(concept_cnt)
        return concepts, concept_cnt

    def userAnalyse(self):
        """


        :return: 根据用户字典，进行简单用户交互信息统计分析
        """
        lcs = []
        s = 0
        maxc, minc = 0, 10000
        for u in self.user_history:
            lc = len(self.user_history[u][0])
            maxc = max(maxc, lc)
            minc = min(minc, lc)
            s += lc
            lcs.append(lc)
        mean = s / self.num_user
        print('user info', self.num_user, mean, maxc, minc)
        return lcs, mean, maxc, minc

    def init_course_dict(self):
        self.course_dict = {}
        """
        course_dict:{course_id:[users], [schools], [teachers], [videos]}
        """
        for c in self.course_set:
            self.course_dict[c] = [[], [], [], []]

    def getDataset(self, uv_file=user_video_file, uc_file=user_course_file):
        """

        :param uv_file: 用户-视频文件
        :param uc_file: 用户-课程文件
        :return: 根据编码，将用户字典、课程字典整理成编码形式
        """
        self.data_encode()
        self.user_dict = {}
        for u in self.user_history:
            u_id = self.user_code[u]
            # print(u_id)
            self.user_dict[u_id] = [[], [], []]
            course_seq = self.user_history[u][0]
            # print(course_seq)

            enroll_time = self.user_history[u][1]
            c_id = [self.course_code[c] for c in course_seq]
            self.user_dict[u_id][0] = c_id
            self.user_dict[u_id][1] = enroll_time

        with open(uv_file, 'r', encoding='utf-8') as f:  # 添加user视频
            for line in f.readlines():
                l = line.strip().split('\t')
                user_id, video_id = l[0], l[1]

                if user_id not in self.user_history or video_id not in self.video_code:
                    continue

                uid = self.user_code[user_id]
                v_id = self.video_code[video_id]
                if v_id not in self.user_dict[uid][2]:
                    self.user_dict[uid][2].append(v_id)

        with open(uc_file, 'r', encoding='utf-8') as file:  # 用户
            for line in file.readlines():
                l = line.strip().split('\t')
                user, course_name = l[0], l[1]
                if user in self.user_history and course_name in self.course_set:
                    if user not in self.course_dict[course_name][0]:
                        self.course_dict[course_name][0].append(user)

        self.course_final_dict = {}
        for c in self.course_dict:
            c_id = self.course_code[c]
            self.course_final_dict[c_id] = [[], [], [], []]
            users, schools, teachers, videos = self.course_dict[c][0], self.course_dict[c][1], self.course_dict[c][2], \
                                               self.course_dict[c][3]
            u_ids = [self.user_code[x] for x in users]
            s_ids = [self.school_code[x] for x in schools]
            t_ids = [self.teacher_code[x] for x in teachers]
            v_ids = [self.video_code[x] for x in videos]

            self.course_final_dict[c_id][0], self.course_final_dict[c_id][1], self.course_final_dict[c_id][2], \
            self.course_final_dict[c_id][3] = u_ids, s_ids, t_ids, v_ids

        return [self.user_dict, self.course_final_dict]

    def data_encode(self):
        """

        :return: 对所有实体进行编码
        """
        u_cnt = 0
        self.user_code = {}
        for u in self.user_history:
            self.user_code[u] = u_cnt
            u_cnt += 1
        print(len(self.user_code))

        c_cnt = 0
        self.course_code = {}
        self.course2name = {}
        self.code2course = {}
        for c in self.course_set:
            self.course_code[c] = c_cnt
            course_name = self.courseid2name[c]
            self.course2name[c_cnt] = course_name
            self.code2course[c_cnt] = c
            c_cnt += 1

        cnt = 0
        self.school_code = {}
        for s in self.schools:
            self.school_code[s] = cnt
            cnt += 1

        cnt = 0
        self.teacher_code = {}
        for t in self.teachers:
            self.teacher_code[t] = cnt
            cnt += 1

        cnt = 0
        self.video_code = {}
        for v in self.videos:
            self.video_code[v] = cnt
            cnt += 1

        # print(len(self.course_code), len(self.school_code), len(self.teacher_code), len(self.video_code))


class dataProcessor(object):
    def __init__(self, data):
        self.data = data
        self.num_users = data.num_user
        self.num_courses = data.num_course
        self.school_num = data.num_school
        self.teacher_num = data.num_teacher

        self.user_history = self.data.user_dict
        self.course_dict = self.data.course_final_dict
        self.course_set = self.data.course_set
        self.negnums = 3
        self.sortUserbehavior()

        # self.user_videos_dict = self.get_user_videos_dict(user_video_file)

        #####users statistic data
        self.cal_user_neighbors()

        #######courses statistic data
        max_users, min_users, average, self.c_maxt, mint, avg_t, _ = self.cal_Course_neighbors()
        self.max_users = max_users//2
        print('max_users', self.max_users)
        print(self.max_seq, self.max_t, self.max_s)

    def generate_user_data(self, user, course_seq, dataset='train'):
        """

        :param user:
        :param course_seq:
        :param dataset:
        :return: 生成每个用户对应的训练数据，包括负采样，padding等步骤
        """

        if dataset == 'train':
            last_idx = -3
        elif dataset == 'valid':
            last_idx = -2
        else:
            last_idx = -1
        target_course = course_seq[last_idx]
        input_seq = course_seq[:last_idx]
        teachers = []
        schools = []
        for c in course_seq:
            s = self.course_dict[c][1]
            t = self.course_dict[c][2]
            teachers.extend(t)
            schools.extend(s)
        teachers = list(set(teachers))
        schools = list(set(schools))
        len_c = np.array([len(input_seq)])
        len_t = np.array([len(teachers)])
        len_s = np.array([len(schools)])
        # target_c_data = self.generate_course_data(user,target_course)
        user = np.array([user])

        ###padding
        input_seq = np.array(self.padding_seq(seq=input_seq, padding_idx=self.num_courses, max_length=self.max_seq)).astype(np.int32)

        teachers = np.array(self.padding_seq(teachers, self.teacher_num, self.max_t)).astype(np.int32)
        schools = np.array(self.padding_seq(schools, self.school_num, self.max_s)).astype(np.int32)

        # print('seq', input_seq.size, 'len teachers', teachers.size, 'len schools', schools.size)

        # user_data = [user, input_seq, teachers, schools, len_c, len_t, len_s]
        # user_data = np.concatenate([user, input_seq, teachers, schools, len_c, len_t, len_s])
        user_data = [user, input_seq, teachers, schools, len_c, len_t, len_s]
        # print('user data length:', user_data.size)

        return user_data

    def generate_course_data(self, user, course, target=False):
        """

        :param user: user id
        :param course_cands: one positive courses and several negative courses
        :return: data in form that:[[courses],[[users for course1],[users for course2],[...]...],[[teachers for course1..],[teachers for c2],...],[school1],school2..]
        [lengths of every users list] [lengths of every teacher list]

        """

        t_users = self.course_dict[course][0]
        len_u = len(t_users)
        if len_u <= self.max_users:
            t_users = t_users + [self.num_users] * (self.max_users - len_u)
        else:
            # t_users = t_users[:self.max_users]
            len_u = self.max_users
            t_users = self.Sample_users(user, t_users, course, self.max_users)

        t_teachers = self.course_dict[course][2]
        if not t_teachers:
            t_teachers = [self.teacher_num]


        c_len_t = len(t_teachers)
        t_teachers = self.padding_seq(t_teachers, self.teacher_num, self.c_maxt)
        t_schools = self.course_dict[course][1]
        # print('origin schools', t_schools)
        if not t_schools:
            t_schools = [self.school_num]
        # print('is teacher', len(t_teachers) > 0)
            #
            # users.append(t_users)
            # teachers.append(t_teachers)
            # schools.append(t_schools)
            # lens_u.append(len_u)
            # lens_t.append(c_len_t)

        users = np.array(t_users)
        # print(users.size)
        teachers = np.array(t_teachers)
        schools = np.array(t_schools)
        course = np.array([course])
        # print(course.size)
        lens_u = np.array([len_u])
        lens_t = np.array([c_len_t])
        if target:
            label = np.array([1])
        else:
            label = np.array([0])
        # courses_data = [course_cands, users, teachers, schools, lens_u, lens_t]
        course_data = np.concatenate([course, users, teachers, schools, lens_u, lens_t, label])

        # c_data = [course_cands, t_users, t_teachers, t_schools, len_u, c_len_t]

        return course_data

    def generate_course_data_pairwise(self, user, course_sets):
        """

        :param user: user id
        :param course_cands: one positive courses and several negative courses
        :return: data in form that:[[courses],[[users for course1],[users for course2],[...]...],[[teachers for course1..],[teachers for c2],...],[school1],school2..]
        [lengths of every users list] [lengths of every teacher list]

        """
        courses = np.array(course_sets)
        course_users = []
        course_teachers = []
        course_schools = []
        lens_u = []
        lens_t = []
        # len_s = []
        for course in course_sets:
            t_users = self.course_dict[course][0]
            len_u = len(t_users)
            if len_u <= self.max_users:
                t_users = t_users + [self.num_users] * (self.max_users - len_u)
            else:
                # t_users = t_users[:self.max_users]
                len_u = self.max_users
                t_users = self.Sample_users(user, t_users, course, self.max_users)

            t_teachers = self.course_dict[course][2]
            if not t_teachers:
                t_teachers = [self.teacher_num]

            c_len_t = len(t_teachers)
            t_teachers = self.padding_seq(t_teachers, self.teacher_num, self.c_maxt)
            t_schools = self.course_dict[course][1]
            # print('origin schools', t_schools)
            if not t_schools:
                t_schools = [self.school_num]
            # users = np.array(t_users).astype(np.int32)
            # teachers = np.array(t_teachers).astype(np.int32)
            # schools = np.array(t_schools).astype(np.int32)
            #
            # len_u = np.array([len_u]).astype(np.int32)
            # len_t = np.array([c_len_t]).astype(np.int32)
            course_users.append(t_users)
            course_schools.append(t_schools)
            course_teachers.append(t_teachers)
            lens_u.append(len_u)
            lens_t.append(c_len_t)
        course_users = np.array(course_users).astype(np.int32)
        course_schools = np.array(course_schools).astype(np.int32)
        course_teachers = np.array(course_teachers).astype(np.int32)
        lens_u = np.array(lens_u).astype(np.int32)
        lens_t = np.array(lens_t).astype(np.int32)

        course_data = [courses, course_users, course_schools, course_teachers, lens_u, lens_t]

        # c_data = [course_cands, t_users, t_teachers, t_schools, len_u, c_len_t]

        return course_data

    def negative_sample_for_user(self, user_id, negnums):
        """
        :param user_id:
        :param negnums:
        :return: 训练时的用户负采样，每个用户对应3个负样本
        """
        user_seqence = set(self.user_history[user_id][0])
        all_courses = set([i for i in range(self.num_courses)])
        neg_courses = []
        sample_set = list(all_courses - user_seqence)

        for j in range(negnums):
            neg = random.choice(sample_set)
            neg_courses.append(neg)

        return neg_courses

    def generate_train_data(self, fast=False):
        train_data = []
        cnt = 0
        # uu = [878, 1840, 1841, 1880, 1888]
        for u in self.user_history:
            course_seq = self.user_history[u][0]
            user_data = self.generate_user_data(user=u, course_seq=course_seq)
            neg_courses = self.negative_sample_for_user(u, self.negnums)
            target_course = course_seq[-3]
            course_cands = [target_course] + neg_courses
            # for i, c in enumerate(course_cands):
            #     if i == 0:
            #         t = True
            #     else:
            #         t = False
            #     course_data = self.generate_course_data(u, c, t)
            #     # train_piece = [user_data, course_data]
            #     train_piece = np.concatenate([user_data, course_data]).astype(np.int32)
            #     # print('train size, user_size, course_size', train_piece.size, user_data.size, course_data.size)
            #     train_data.append(train_piece)
            #     cnt += 1
            #     if fast:
            #         if cnt >= 500:
            #             return train_data
            course_data = self.generate_course_data_pairwise(u, course_cands)
            train_piece = [user_data, course_data]
            train_data.append(train_piece)
            cnt += 1
            if fast:
                if cnt >= 500:
                    return train_data
                # print('train cnt', cnt)


        return train_data

    def generate_valid_and_test_data(self, fast=False):
        user_negative_cands = self.generate_users_negative_candicates()
        test_data = []
        valid_data = []
        cnt = 0
        for u in self.user_history:
            course_seq = self.user_history[u][0]
            user_data_test = self.generate_user_data(user=u, course_seq=course_seq, dataset='test')
            user_data_valid = self.generate_user_data(user=u, course_seq=course_seq, dataset='valid')
            neg_courses = user_negative_cands[u]
            target_course_test = course_seq[-1] # test target

            target_course_valid = course_seq[-2] # valid target
            course_cands = [target_course_test] + [target_course_valid] + neg_courses
            for i, c in enumerate(course_cands):
                # print(c)
                if i == 0 or i == 1:
                    target = True
                    if i == 0:
                        course_data_test = self.generate_course_data(u, c, target)
                        test_piece = np.concatenate([user_data_test, course_data_test]).astype(np.int32)
                        test_data.append(test_piece)
                        cnt += 1
                        if fast:
                            if cnt >= 1000:
                                return valid_data, test_data
                    elif i == 1:
                        course_data_valid = self.generate_course_data(u, c, target)
                        valid_piece = np.concatenate([user_data_valid, course_data_valid]).astype(np.int32)
                        valid_data.append(valid_piece)
                        cnt += 1
                        if fast:
                            if cnt >= 1000:
                                return valid_data, test_data

                else:
                    target = False
                    course_data = self.generate_course_data(u, c, target)

                    test_piece = np.concatenate([user_data_test, course_data]).astype(np.int32)
                    valid_piece = np.concatenate([user_data_valid, course_data]).astype(np.int32)
                    test_data.append(test_piece)
                    valid_data.append(valid_piece)
                    cnt += 1
                    if fast:
                        if cnt >= 1000:
                            return valid_data, test_data
                # print('train cnt', cnt)


        return valid_data, test_data

    def generate_valid_and_test_data_pairwise(self, fast=False):
        user_negative_cands = self.generate_users_negative_candicates()
        test_data = []
        valid_data = []
        cnt = 0
        for u in self.user_history:
            course_seq = self.user_history[u][0]
            user_data_test = self.generate_user_data(user=u, course_seq=course_seq, dataset='test')
            user_data_valid = self.generate_user_data(user=u, course_seq=course_seq, dataset='valid')
            neg_courses = user_negative_cands[u]


            target_course_valid = course_seq[-2] # valid target
            course_valid_cands = [target_course_valid] + neg_courses
            course_valid_data = self.generate_course_data_pairwise(u, course_valid_cands)
            valid_piece = [user_data_valid, course_valid_data]

            target_course_test = course_seq[-1]  # test target
            course_test_cands = [target_course_test] + neg_courses
            course_test_data = self.generate_course_data_pairwise(u, course_test_cands)
            test_piece = [user_data_test, course_test_data]

            valid_data.append(valid_piece)
            test_data.append(test_piece)
            cnt += 1
            print(cnt)
            if fast:
                if cnt >= 500:
                    return [valid_data, test_data]

        return [valid_data, test_data]

    def cal_user_neighbors(self):
        """

        :return: the statistic of users neighbors, like the max sequence length, max teachers number, and the sparsity
        """

        max_seq = 0
        max_t = 0
        max_s = 0

        for u in self.user_history:
            uid = u
            course_sequence = self.user_history[u][0]
            max_seq = max(max_seq, len(course_sequence))
            teachers = []
            schools = []
            for c in course_sequence:
                s = self.course_dict[c][1]
                t = self.course_dict[c][2]
                teachers.extend(t)
                schools.extend(s)
            teachers = list(set(teachers))
            schools = list(set(schools))
            max_t = max(len(teachers), max_t)
            max_s = max(len(schools), max_s)
        self.max_seq = max_seq
        self.max_s = max_s
        self.max_t = max_t

    def cal_Course_neighbors(self):
        """

        :return: 课程相关的统计信息，包括最大最小用户数量，平均用户数量
        """
        max_users = 0
        min_users = 100
        maxt, mint = 0, 1000
        average, avg_t = 0, 0
        num_users_list = []
        # print(len(self.course_dict))
        for c in self.course_dict:
            all_c = self.course_dict[c]
            num_users = len(all_c[0])
            num_users_list.append(num_users)
            average += num_users
            max_users = max(num_users, max_users)
            min_users = min(num_users, min_users)

            num_teachers = len(all_c[2])
            maxt = max(num_teachers, maxt)
            mint = min(num_teachers, mint)
            avg_t += num_teachers
        average /= len(self.course_dict)
        avg_t /= len(self.course_dict)
        # sns.distplot(num_users_list)
        # plt.show()
        # self.course_max_t = maxt
        print('Max users for course:', max_users)

        return max_users, min_users, average, maxt, mint, avg_t, num_users_list

    def sortUserbehavior(self):
        self.user_session = {}
        def getSemester(enrolltime):
            """

            :param enrolltime: 上课时间
            :return: 对应学期标号
            """
            time = datetime.strptime(enrolltime, '%Y-%m-%d %H:%M:%S')
            if time >= datetime.strptime('2015-06-23 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2015-08-31 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 1
            elif time >= datetime.strptime('2015-09-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2016-02-29 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 2
            elif time >= datetime.strptime('2016-03-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2016-08-31 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 3
            elif time >= datetime.strptime('2016-09-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2017-02-28 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 4
            elif time >= datetime.strptime('2017-03-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2017-08-31 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 5
            elif time >= datetime.strptime('2017-09-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2018-02-28 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 6
            elif time >= datetime.strptime('2018-03-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2018-08-31 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 7
            elif time >= datetime.strptime('2018-09-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2019-02-28 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 8
            elif time >= datetime.strptime('2019-03-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2019-08-31 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 9
            elif time >= datetime.strptime('2019-09-01 00:00:00', '%Y-%m-%d %H:%M:%S') and time <= datetime.strptime(
                    '2019-11-13 23:59:59', '%Y-%m-%d %H:%M:%S'):
                return 10
        def getSeqInoder(enrolltime, course_order):
            x = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in enrolltime]
            zipped = zip(x, course_order)
            # print(zipped)
            c_inorder = sorted(zipped, key=lambda x: x[0])
            seq = [x[1] for x in c_inorder]
            enroll_time = [str(x[0]) for x in c_inorder]
            return seq, enroll_time

        for u in self.user_history:
            self.user_session[u] = collections.defaultdict(list)
            course_order = self.user_history[u][0]
            enroll_time = self.user_history[u][1]
            # print(enroll_time)
            user_sequence, enroll_time = getSeqInoder(enroll_time, course_order)
            session = [getSemester(t) for t in enroll_time]
            for i in range(len(session)):
                self.user_session[u][session[i]].append(user_sequence[i])

            self.user_history[u][0] = user_sequence

    def Sample_users(self, uid, users, course, num):
        """
        :param users: courses interactives users, sample num users to reduce complexity
        :return: sampled(cut down) users
        """
        if uid in users:
            users.remove(uid)
        watch_radio = []
        for u in users:
            u_videos = self.user_history[u][2]
            course_videos = self.course_dict[course][3]
            inter = list(set(u_videos) & set(course_videos))
            lu = len(inter)
            lc = len(course_videos)
            radio = lu / lc
            watch_radio.append(radio)
        z = list(zip(users, watch_radio))
        a = sorted(z, key=lambda x: x[1], reverse=True)
        sampled_users = [i[0] for i in a][:num]
        # sampled_users = sampled_users[:num]

        return sampled_users

    def padding_seq(self, seq, padding_idx, max_length):
        l = len(seq)
        if l >= max_length:
            return seq
        seq = seq + [padding_idx] * (max_length - l)
        return seq

    def generate_users_negative_candicates(self, neg_nums=99):
        """

        :param neg_nums:
        :return:生成测试与验证时的负样本，每个用户对应99个
        """
        self.user_negative_courses = {}
        all_courses = [c for c in range(self.num_courses)]
        for u in self.user_history:
            negs = []
            sequence = self.user_history[u][0]
            candicates = list(set(all_courses)-set(sequence))

            for j in range(neg_nums):
                neg = random.choice(candicates)
                while neg in negs:
                    neg = random.choice(candicates)
                negs.append(neg)
            negs = list(set(negs))
            self.user_negative_courses[u] = negs
            # print(len(negs))
        return self.user_negative_courses

    def showPaddingInfo(self):
        print('max sequence length', self.max_seq)
        print('user max t', self.max_t)
        print('user max s', self.max_s)
        print('max users', self.max_users)
        print('course max t', self.c_maxt)


# uf = 15
# cf = 10
# # generator = dataGenerator(uf, cf)
# # with open('./data_process/generator_uf%d_cf%d.pkl' % (generator.user_filter, generator.course_filter), 'wb') as f:
# #     pickle.dump(generator, f)
# with open('./data_process/generator_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
#     generator = pickle.load(f)
# # with open('./data_process/processor_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
# #     processor = pickle.load(f)
# #
# # valid_test = processor.generate_valid_and_test_data_pairwise()
# # with open('./data_process/valid_test_data_uf%d_cf%d.pkl' % (generator.user_filter, generator.course_filter), 'wb') as f:
# #     pickle.dump(valid_test, f)
#
# # generator = dataGenerator(user_filter=uf, course_filter=cf)
# # generator.getDataset()
#
# print(generator.num_user, generator.num_course)
# print(generator.user_dict[0][0])
# print(generator.user_dict[1][0])

# num_course = generator.num_course
def get_seq_train_test_data(user_history, num_course, user_neg_course_dict, seq_len=5, target_len=2,):
    train_data = []
    test_data = []
    for u in user_history:
        # if u>0: break
        user_seq = user_history[u][0]
        neg_courses = user_neg_course_dict[u]
        test_cands = [user_seq[-1]] + neg_courses
        test_data.append([np.array(user_seq[-(seq_len + 1):-1]), np.array(test_cands)])
        for input_seq, target_seq in slide_win(user_seq[:-1], win_size=seq_len, target_len=target_len):  # 留一法
            negs = negative_sample_for_user(num_course, user_seq, negnums=3)
            target_seq += negs
            train_data.append([np.array(input_seq), np.array(target_seq)])
    return train_data, test_data


def negative_sample_for_user(num_courses, user_sequence, negnums):
        """
        :param user_id:
        :param negnums:
        :return: 训练时的用户负采样，每个用户对应3个负样本
        """
        all_courses = set([i for i in range(num_courses)])
        neg_courses = []
        sample_set = list(all_courses - set(user_sequence))

        for j in range(negnums):
            neg = random.choice(sample_set)
            neg_courses.append(neg)

        return neg_courses

def slide_win(sequence,  win_size, target_len, step=1):
    start = 0
    end = win_size
    while end < len(sequence)-target_len + 1:
        yield sequence[start:end], sequence[end:end+target_len]
        start +=step
        end += step

if __name__ == '__main__':
    uf, cf = 10, 10
    with open('./data_process/generator_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
        generator = pickle.load(f)
    # generator = dataGenerator(uf, cf)
    # with open('./data_process/generator_uf%d_cf%d.pkl' % (uf, cf), 'wb') as f:
    #     pickle.dump(generator, f)

    processor = dataProcessor(generator)
    processor.generate_users_negative_candicates()

    # train_data, test_data = get_seq_train_test_data(generator.user_dict, generator.num_course, user_neg_course_dict=processor.user_negative_courses)
    print('num users', generator.num_user)
    print('num courses', generator.num_course)

    # print(generator.user_dict[0][0])
    # for i in range(len(train_data)):
    #     print(train_data[i],'\n')
    # print(test_data[0])
    cnt = 0
    for u in processor.user_session:
        cnt += 1
        if cnt >10:break
        print(processor.user_session[u])

# train_data, test_data = get_seq_train_test_data(generator.user_dict)
# print(len(train_data))
# print(len(test_data))
# train_iter = DataLoader(train_data, batch_size=2, shuffle=False)

# for i,v in enumerate(train_iter) :
#     print(v[0], v[1])
# with open('./data_process/generator_uf%d_cf%d.pkl' % (generator.user_filter, generator.course_filter), 'wb') as f:
#     pickle.dump(generator, f)
# # generator.showInfo()
# dataset = generator.getDataset()
# processor = dataProcessor(data=generator)
# #
# with open('./data_process/processor_uf%d_cf%d.pkl' % (generator.user_filter, generator.course_filter), 'wb') as f:
#     pickle.dump(processor, f)
# processor.showPaddingInfo()
# valid_test = processor.generate_valid_and_test_data_pairwise()
# with open('./data_process/valid_test_data_uf%d_cf%d.pkl' % (generator.user_filter, generator.course_filter), 'wb') as f:
#     pickle.dump(valid_test, f)
# #
# # user_negative_cands = precessor.generate_users_negative_candicates()
# train = precessor.generate_train_data(fast=False)
# print(len(train))
# print(train[2])
#
# train_iter = DataLoader(train, batch_size=128, shuffle=True)
# for i,v in enumerate(train_iter):
#     print(v)
# print(len(valid_data))
# print(len(test_data))
# n_us, mean_u, max_u, min_u = generator.getCourses(user_course_file)
# print(mean_u, max_u, min_u)
#
# lcs, mean, maxc, minc = generator.userAnalyse()
# generator.reloadUsers()
# _, _, _, _ = generator.userAnalyse()
