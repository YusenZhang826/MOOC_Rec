from GenerateDataset import dataProcessor, dataGenerator
from MyModel import *
from torch.utils.data import DataLoader
import pickle
import torch
import torch.optim as optim
from evaluate import *
import time


device = torch.device('cpu')
uf = 15
cf = 10
# generator = dataGenerator(user_filter=uf, course_filter=cf)
with open('./data_process/generator_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
    generator = pickle.load(f)
with open('./data_process/processor_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
    processor = pickle.load(f)

generator.showInfo()
processor.showPaddingInfo()
print('Load train data')
train = processor.generate_train_data(fast=False)
print('train data num', len(train))

print('Load valid and test data')
with open('./data_process/valid_test_data_uf%d_cf%d.pkl' % (uf, cf), 'rb') as f:
    valid_test = pickle.load(f)
valid, test = valid_test[0], valid_test[1]
# valid = valid[:100]
valid_iter = DataLoader(valid, batch_size=1)
test_iter = DataLoader(test, batch_size=1)

# valid, test =processor.generate_valid_and_test_data_pairwise()
#
# valid_iter = DataLoader(valid, batch_size=1, shuffle=False)

batch = 256
batch_num = len(train) // batch + 1
Epoch = 20
lr = 0.03
l2 = 1e-5

train_iter = DataLoader(train, batch_size=batch, shuffle=True)
# print(type(train_iter))
train_data = [x for x in train_iter]
# print(train_data[0])
num_user = generator.num_user
num_course = generator.num_course
num_teacher = generator.num_teacher
num_school = generator.num_school

gcn = myGCN(num_user, num_course, num_teacher, num_school).to(device)

HIN = myHIN(num_user, num_course, num_teacher, num_school).to(device)



def train_model(model, train_iter, Epoch, valid_iter, test_iter, lr):
    print('Start train')
    model.train()
    for epoch in range(Epoch):
        if epoch+1 >= 10:
            lr *= 0.8
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=l2)
        t1 = time.time()
        total_loss = 0
        for i, v in enumerate(train_iter):
            user_data, course_data = v[0], v[1]
            out = model(user_data, course_data)
            (targets_prediction, negatives_prediction) = torch.split(out, [1, 3], dim=1)
            loss = Binary_CrossEntropy_loss(targets_prediction, negatives_prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     print(name, param.grad, )
            total_loss += loss.item()
        total_loss /= batch_num
        t2 = time.time()
        print("Epoch:%d,loss:%.4f,time:%.4f " % (epoch + 1, total_loss, (t2 - t1)))

        if (epoch + 1) % 1 == 0:
            tt1 = time.time()
            (hr5, ndcg5, hr10, ndcg10, Mrr5, Mrr10) = test_model(model, valid_iter)
            tt2 = time.time()
            print(
                "Valid results : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MRR5 = %.4f, MRR10 = %.4f, Test time: %.4f" % (
                    hr5, ndcg5, hr10, ndcg10, Mrr5, Mrr10, tt2 - tt1))
            model.train()

    (hr5, ndcg5, hr10, ndcg10, mrr5, mrr10) = test_model(model, test_iter)
    print(
        "Test results : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MRR5 = %.4f, MRR10 = %.4f" % (
            hr5, ndcg5, hr10, ndcg10, mrr5, mrr10))

def test_model(model, test_iter):
    hits5, hits10, ndcgs5, ndcgs10, mrr5, mrr10 = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, v in enumerate(test_iter):
            # if i>5:break
            user_data = v[0]
            course_data = v[1]

            predicts = model(user_data, course_data)
            predicts = predicts.cpu().numpy().flatten()
            hr5, ndcg5, hr10, ndcg10, mr5, mr10 = evaluation(predicts)
            hits5.append(hr5)
            ndcgs5.append(ndcg5)
            hits10.append(hr10)
            ndcgs10.append(ndcg10)
            mrr5.append(mr5)
            mrr10.append(mr10)


        final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_mrr5, final_mrr10 = np.array(hits5).mean(), np.array(ndcgs5).mean(), \
           np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(mrr5).mean(), np.array(mrr10).mean()

        return (final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_mrr5, final_mrr10)



train_model(gcn, train_iter, Epoch, valid_iter, test_iter, lr)
# for i in range(5):
#     print(HIN(train_data[i][0], train_data[i][1]))