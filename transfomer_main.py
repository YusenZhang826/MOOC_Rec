from Transformer4Rec import *
import random
from torch.utils.data import DataLoader
import pickle
import torch
import torch.nn.utils as utils
import torch.optim as optim
from evaluate import *
import time
from GenerateDataset import dataGenerator, dataProcessor

with open('./data_process/all_subjects_class', 'rb') as f:
    fileload = pickle.load(f)
[course_name_map, name2course, first_class_subject, course_subject_map, subject2code] = fileload


def get_seq_train_test_data(user_history, num_course, user_neg_course_dict, seq_len=5, target_len=1, ):
    train_data = []
    test_data = []
    for u in user_history:
        # if u>1: break
        user_seq = user_history[u][0]
        neg_courses = user_neg_course_dict[u]
        tgt = user_seq[-1]

        test_seq = user_seq[-(seq_len + 1):-1]
        neg_courses.insert(0, tgt)
        test_data.append([np.array(test_seq), np.array(neg_courses),
                          get_course_subjects(test_seq, code2course, subject2code, course_subject_map),
                          get_course_subjects(neg_courses, code2course, subject2code, course_subject_map)])

        for input_seq, target_seq in slide_win(user_seq[:-1], win_size=seq_len, target_len=target_len):  # 留一法
            input_subjects = get_course_subjects(input_seq, code2course, subject2code, course_subject_map)
            negs = negative_sample_for_user(num_course, user_seq, negnums=3)
            target_seq += negs

            target_subjects = get_course_subjects(target_seq, code2course, subject2code, course_subject_map)
            train_data.append([np.array(input_seq), np.array(target_seq), input_subjects, target_subjects])
    return train_data, test_data


def get_course_subjects(sequence, code2course, subject2code, course_subject_map):
    """

    :param sequence: course sequence code id, length=L
    :param code2course: course code id to real id
    :param subject2code: subject name to code
    :param course_subject_map: course real code to subject name
    :return: each course's main subjects L*[subject1, subject2]
    """
    L = len(sequence)
    ans = np.zeros(shape=(L, 2))
    for i in range(L):
        course_id = code2course[sequence[i]]
        subjects_name = course_subject_map[course_id][0]
        subjects_id = [subject2code[sub] for sub in subjects_name]
        if len(subjects_id) < 2:
            subjects_id.append(subjects_id[0])
        ans[i][0] = subjects_id[0]
        ans[i][1] = subjects_id[1]
    return ans


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


def slide_win(sequence, win_size, target_len, step=1):
    if len(sequence) < win_size:
        return sequence
    start = 0
    end = win_size
    while end < len(sequence) - target_len + 1:
        yield sequence[start:end], sequence[end:end + target_len]
        start += step
        end += step


print(torch.cuda.is_available())
uf, cf = 5, 5
# with open('./data_process/generator_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
#     generator = pickle.load(f)
# with open('./data_process/generator_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
#     generator = pickle.load(f)
generator = dataGenerator(uf, cf)
code2course = generator.code2course

num_course = generator.num_course
processor = dataProcessor(generator)
processor.generate_users_negative_candicates()

train_data, test_data = get_seq_train_test_data(processor.user_history, generator.num_course,
                                                user_neg_course_dict=processor.user_negative_courses)
# train_data = train_data[:10000]
# test_data = test_data[:1000]

print('num_train', len(train_data))
print('num_test', len(test_data))
print(generator.user_dict[0][0])
# print(train_data)
# print(test_data[0])

Epoch = 30
lr = 0.003
l2 = 1e-9
batch_size = 512
batch_num = len(train_data) // batch_size + 1
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=1, shuffle=True)
# for i, v in enumerate(train_iter):
#     if i > 0: break
#     print(v[0], '\n', v[1], '\n', v[2], '\n', v[3])

model = FM_Transoformer(num_course).cuda()
# model = Transformer4Rec(num_course).cuda()
# model = FM(num_course).cuda()
# model = LSTM(num_course).cuda()

# optimizer = optim.Adam(model.parameters(), lr=0.001)
def test_model(model, test_iter):
    hits5, hits10, ndcgs5, ndcgs10, mrr5, mrr10 = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, v in enumerate(test_iter):
            # if i>5:break
            # input_seq = v[0].long().cuda()
            # tgt_seq = v[1].long().cuda()
            input_seq, tgt_seq, seq_subject, tgt_subject = v[0].long().cuda(), v[1].long().cuda(), v[2].long().cuda(), \
                                                           v[3].long().cuda()
            predicts = model(input_seq, tgt_seq, seq_subject, tgt_subject)
            predicts = predicts.cpu().numpy().flatten()
            sorted_idx = predicts.argsort().tolist()[::-1][:10]
            input_seq = input_seq.cpu().numpy().flatten().tolist()
            tgt_seq = tgt_seq.cpu().numpy().flatten().tolist()
            hr5, ndcg5, hr10, ndcg10, mr5, mr10 = evaluation(predicts, 0)

            # print('input sequence:', [generator.course2name[j] for j in input_seq], '--->', 'target course', generator.course2name[tgt_seq[0]])
            # print('recommendation list:', [generator.course2name[tgt_seq[j]] for j in sorted_idx])
            # print('---------------------------------------------------------------------------------------------------------------')
            hits5.append(hr5)
            ndcgs5.append(ndcg5)
            hits10.append(hr10)
            ndcgs10.append(ndcg10)
            mrr5.append(mr5)
            mrr10.append(mr10)


        final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_mrr5, final_mrr10 = np.array(hits5).mean(), np.array(ndcgs5).mean(), \
           np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(mrr5).mean(), np.array(mrr10).mean()

        return (final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_mrr5, final_mrr10)

print('Start train')
print('uf=', uf, 'cf=', cf)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
model.train()
for epoch in range(Epoch):
    # if epoch + 1 >= 10:
    #     lr *= 0.9

    t1 = time.time()
    total_loss = 0
    for i, v in enumerate(train_iter):
        # input_seq, tgt_seq = v[0].long().cuda(), v[1].long().cuda()
        input_seq, tgt_seq, seq_subject, tgt_subject = v[0].long().cuda(), v[1].long().cuda(), v[2].long().cuda(), v[3].long().cuda()
        out = model(input_seq, tgt_seq, seq_subject, tgt_subject)
        (targets_prediction, negatives_prediction) = torch.split(out, [1, 3], dim=1)
        loss = Binary_CrossEntropy_loss(targets_prediction, negatives_prediction)
        optimizer.zero_grad()

        loss.backward()
        utils.clip_grad_value_(model.parameters(), clip_value=1e-4)
        # utils.clip_grad_norm_(model.parameters(), max_norm=20,)
        # for name, param in model.named_parameters():
        #     print('param', name, param, '\n', 'grad', param.grad)
        optimizer.step()

        total_loss += loss.item()
    total_loss /= batch_num
    t2 = time.time()
    print("Epoch:%d,loss:%.4f,time:%.4f " % (epoch + 1, total_loss, (t2 - t1)))
    if (epoch + 1) % 5 == 0:
            tt1 = time.time()
            (hr5, ndcg5, hr10, ndcg10, mrr5, mrr10) = test_model(model, test_iter)
            tt2 = time.time()
            print(
                "Test results : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MRR5 = %.4f, MRR10 = %.4f， time=%.2f" % (
                    hr5, ndcg5, hr10, ndcg10, mrr5, mrr10, tt2-tt1))
            model.train()
