import numpy as np
import json
import numpy as np
import pickle
import collections
import random
from torch.utils.data import DataLoader
from datetime import datetime

userfile = './data_file/user.json'


def getSemester(enrolltime):
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


with open(userfile, 'rb') as file:
    # dict = collections.defaultdict(list)
    times = []
    user_session = {}
    for line in file.readlines():
        if len(user_session) > 100: break
        dic = json.loads(line)
        user_id = dic['name']
        course_order = dic['course_order']  # delete users whose actions less than 5
        enroll_time = dic['enroll_time']
        user_session[user_id] = collections.defaultdict(list)

        course_seq, enroll_time = getSeqInoder(enroll_time, course_order)
        session = [getSemester(t) for t in enroll_time]
        course_session = sorted(zip(course_seq, enroll_time, session), key=lambda x: x[1])
        for i in range(len(session)):
            user_session[user_id][session[i]].append((course_seq[i], enroll_time[i]))
        x = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in enroll_time]
        # times.extend(x)
    # return user_session
for u in user_session:
    print(user_session[u])
# times = sorted(times)
# for i in range(10):
#     print(times[i])
# print(len(times))
# print(times[-1])
