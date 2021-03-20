# coding=UTF-8
import json
import numpy as np
import pickle
import collections
import random

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
concept_field_file = './data_file/concept-field.json'
course_concept_file = './data_file/course-concept.json'


def get_concept_subject_map(concept_file='./data_file/concept.json'):
    with open(concept_file, 'rb') as file:
        # dict = collections.defaultdict(list)
        concept_subject_dict = collections.defaultdict(list)

        for line in file.readlines():
            # if len(concept_subject_dict) > 10: break
            dic = json.loads(line)
            # print(dic)
            concept_id = dic['id']
            if 'explanation' not in dic:
                continue
            explain = dic['explanation'][3:]  # 去掉前缀“学科: ”
            sapce_idx = explain.index(' ')
            subject = explain[:sapce_idx]  # 得到学科级别
            fine_grain_sub = subject.strip().split('_')  # 得到层次化学科
            for sub in fine_grain_sub:
                concept_subject_dict[concept_id].append(sub)
                if len(concept_subject_dict[concept_id]) >= 3:
                    break
        return concept_subject_dict


def get_course_concept_map(concept_file='./data_file/course-concept.json'):
    # with open(concept_file, 'rb') as file:
    # dict = collections.defaultdict(list)
    course_concept_dict = collections.defaultdict(set)
    with open(concept_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            # if len(course_concept_dict) > 10: break
            l = line.strip().split('\t')
            course_id, concept = l[0], l[1]
            course_concept_dict[course_id].add(concept)

        return course_concept_dict


def get_course_name_map(course_file='./data_file/course.json'):
    with open(course_file, 'rb') as file:
        # dict = collections.defaultdict(list)
        course_name_dict = dict()
        name2course = dict()
        all_names = set()
        repetition = collections.defaultdict(int)

        for line in file.readlines():
            # if len(concept_subject_dict) > 10: break
            dic = json.loads(line)
            # print(dic)
            course_id = dic['id']
            course_name = dic['name'].replace(" ", "")

            if course_name in all_names:
                repetition[course_name] += 1
                course2_name = course_name + '*'*(repetition[course_name])
                name2course[course2_name] = course_id
                course_name_dict[course_id] = course2_name
            else:
                name2course[course_name] = course_id
                course_name_dict[course_id] = course_name
            all_names.add(course_name)



    return course_name_dict, name2course


def filtconcepts(concepts):
    """

    :param concepts: 课程对应的概念
    :return: 对概念筛选，保留出现频率最高的两种一级学科对应的概念
    """
    counter = collections.defaultdict(int)
    for c in concepts:
        subject = c.split('_')[-1]
        counter[subject] += 1
    sorted_subjects = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 按频次排序
    if len(sorted_subjects) >= 2 :
        top_subjects = [sorted_subjects[0][0], sorted_subjects[1][0]]  # 取频次前二的一级学科
    else:
        return concepts

    filter_concepts = []
    for concept in concepts:
        sub = concept.split('_')[-1]
        if sub == top_subjects[0] or sub == top_subjects[1]:
            filter_concepts.append(concept)
    return filter_concepts


def get_course_subject_map(concept_subject_map, course_concept_map, course_name_map):
    other_class = '其他'  # 对应没有概念属性的课程，归为其他类别
    course_subject_map = dict()  # 课程所属学科
    first_class_subject = collections.defaultdict(set)  # 一级学科课程群
    second_class_subject = collections.defaultdict(set)  # 二级学科课程群
    third_class_subject = collections.defaultdict(set)  # 三级学科课程群
    for course in course_name_map:
        course_subject_map[course] = [set(), set(), set()]  # 记录课程分别所属的三级学科
        name = course_name_map[course]

        if course not in course_concept_map:  # 其他类别课程
            if name in ["微积分1（CalculusI）（2019春）", "线性代数（自主模式）", "高等数学（自主模式）", ]:
                course_subject_map[course][0].add("数学")
                first_class_subject["数学"].add(course)
            elif name in ["软件工程（自主模式）", "软件即服务(SaaS)-第2部分"]:
                course_subject_map[course][0].add("计算机科学技术")
                first_class_subject["计算机科学技术"].add(course)

                course_subject_map[course][1].add("计算机科学技术-软件工程")
                second_class_subject["计算机科学技术-软件工程"].add(course)

            elif name in ["藏语言文字学概论（2019春）", "藏语言文字学概论"]:
                course_subject_map[course][0].add("语言学")
                first_class_subject["语言学"].add(course)

                course_subject_map[course][1].add("语言学-民族语言学")
                second_class_subject["语言学-民族语言学"].add(course)

            elif name in ["管理学原理（2019春）"]:
                course_subject_map[course][0].add("管理科学技术")
                first_class_subject["管理科学技术"].add(course)

            elif name in ["大学化学(2019春)"]:
                course_subject_map[course][0].add("化学")
                first_class_subject["化学"].add(course)

            elif name in ["英文写作指导——写作入门（2019春）", "英语语音（自主模式）", "大学英文写作（上）（2018秋）", "对外汉语（自主模式）"]:
                course_subject_map[course][0].add("语言学")
                first_class_subject["语言学"].add(course)
                yuyan = ["语言学-计算语言学", "语言学-辞书学", "语言学-语法学", "语言学-语义学、词汇学", "语言学-语音学", "语言学-理论语言学", "语言学-修辞学",
                         "语言学-音韵学", "语言学-文字学", "语言学-民族语言学", "语言学-社会语言学", "语言学-训诂学", "语言学-方言学", ]
                for y in yuyan:
                    course_subject_map[course][1].add(y)
                    second_class_subject[y].add(course)

            elif name in ["教育环境中跨文化交流的秘密（2019春）"]:
                course_subject_map[course][0].add("教育学")
                first_class_subject["教育学"].add(course)

                educates = ["教育学-教育基本理论", "教育学-比较教育", "教育学-职业技术教育", "教育学-高等教育", "教育学-课程与教学",
                            "教育学-教师教育", "教育学-教育技术", "教育学-初等中等教育", "教育学-外国教育史", "教育学-教育管理",
                            "教育学-特殊教育", "教育学-成人教育", "教育学-中国教育史", "教育学-学前教育", "教育学-教育经济", ]
                for edu in educates:
                    course_subject_map[course][1].add(edu)
                    second_class_subject[edu].add(course)
            else:
                course_subject_map[course][0].add(other_class)
                first_class_subject[other_class].add(course)

        else:
            concepts = course_concept_map[course]  # 课程对应的所有概念 set集合
            concepts = filtconcepts(concepts)
            for concept in concepts:
                subjects = concept_subject_map[concept]  # 概念对应的所有层次化学科分类
                # if len(subjects) == 3:
                for i, sub in enumerate(subjects):
                    if i == 0:  # 一级学科
                        course_subject_map[course][0].add(sub)
                        first_class_subject[sub].add(course)

                    elif i == 1:  # 二级学科
                        second_sub = subjects[0] + '-' + sub
                        course_subject_map[course][1].add(second_sub)
                        second_class_subject[second_sub].add(course)
                    elif i == 2:  # 三级学科
                        third_sub = subjects[0] + '-' + subjects[1] + '-' + sub
                        course_subject_map[course][2].add(third_sub)
                        third_class_subject[third_sub].add(course)

    return course_subject_map, first_class_subject, second_class_subject, third_class_subject


course_name_map, name2course = get_course_name_map()  # 全部课程对应名字
print('num course', len(course_name_map), len(name2course))

concept_subject_map = get_concept_subject_map()
print(len(concept_subject_map))

course_concept_map = get_course_concept_map()
print(len(course_concept_map))

course_subject_map, first_class_subject, second_class_subject, third_class_subject = get_course_subject_map(
    concept_subject_map, course_concept_map, course_name_map)
print(len(course_subject_map),
      len(first_class_subject),
      len(second_class_subject),
      len(third_class_subject))

# print(first_class_subject)
first_class = [f for f in first_class_subject]
second_class = [s for s in second_class_subject]
third_class = [t for t in third_class_subject]
all_subjects = [first_class, second_class, third_class]
# with open('./data_process/all_subjects_class', 'wb') as f:
#     pickle.dump(all_subjects, f)

print(first_class)
print(second_class)
print(third_class)

max_first_len = 0
min_first_len = 1000
avg_first_len = 0
for f in first_class_subject:
    n = len(first_class_subject[f])
    max_first_len = max(max_first_len, n)
    min_first_len = min(min_first_len, n)
    avg_first_len += n
avg_first_len /= len(first_class_subject)

max_second_len = 0
min_second_len = 1000
avg_second_len = 0
for f in second_class_subject:
    n = len(second_class_subject[f])
    max_second_len = max(max_second_len, n)
    min_second_len = min(min_second_len, n)
    avg_second_len += n
avg_second_len /= len(second_class_subject)

max_third_len = 0
min_third_len = 1000
avg_third_len = 0
for f in third_class_subject:
    n = len(third_class_subject[f])
    max_third_len = max(max_third_len, n)
    min_third_len = min(min_third_len, n)
    avg_third_len += n
avg_third_len /= len(third_class_subject)

print(max_first_len, max_second_len, max_third_len)
print(min_first_len, min_second_len, min_third_len)
print(avg_first_len, avg_second_len, avg_third_len)

course1, course2, course3 = 0, 0, 0
for c in course_subject_map:
    if len(course_subject_map[c][0]) > 2:
        print(course_name_map[c], course_subject_map[c][0])
    course1 += len(course_subject_map[c][0])
    course2 += len(course_subject_map[c][1])
    course3 += len(course_subject_map[c][2])
course1 /= len(course_subject_map)
course2 /= len(course_subject_map)
course3 /= len(course_subject_map)
# print(course1, course2, course3)

# print([course_name_map[i] for i in third_class_subject["机械工程-物料搬运机械-起重机械"]])
# print(course_subject_map["C_course-v1:TsinghuaX+AP000002X+2019_T1"])
print(course_subject_map[name2course["医学寄生虫学(2019春)"]])
# for f in first_class_subject:
#     if len(first_class_subject[f]) == max_first_len:
#         print(f, [course_name_map[i] for i in first_class_subject[f]])

print([course_name_map[i] for i in first_class_subject["农学"]])
subject2code = {}
cnt = 0
for subject in first_class_subject:
    subject2code[subject] = cnt
    cnt += 1
print(subject2code)
file2save = [course_name_map, name2course, first_class_subject, course_subject_map, subject2code]
with open('./data_process/all_subjects_class', 'wb') as f:
    pickle.dump(file2save, f)