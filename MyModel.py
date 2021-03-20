import torch
import torch.nn as nn
import torch.optim as optim
import math

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cpu')
class myGCN(nn.Module):
    def __init__(self, num_user, num_course, num_teacher, num_school, num_dim=16,
                 lstm_hidden=16):
        super(myGCN, self).__init__()

        self.num_dim = num_dim
        self.user_embed = nn.Embedding(num_user + 1, num_dim, padding_idx=num_user)
        self.course_embed = nn.Embedding(num_course + 1, num_dim, padding_idx=num_course)
        self.teacher_embed = nn.Embedding(num_teacher + 1, num_dim, padding_idx=num_teacher)
        self.school_embed = nn.Embedding(num_school + 1, num_dim, padding_idx=num_school)

        self.mlp = nn.Sequential(nn.Linear(num_dim, 1))
        self.init_param()

    def init_param(self):
        for param in self.parameters():
            nn.init.normal_(param, 0, 0.001)

    def forward(self, user_data, course_data):
        ##### load user data #########
        user_idx = user_data[0].long().to(device)
        user_sequence = user_data[1].long().to(device)
        user_teachers = user_data[2].long().to(device)
        user_school = user_data[3].long().to(device)
        user_len_seq = user_data[4].float() .to(device)      # (b, 1)
        user_len_teacher = user_data[5].float().to(device)  # (b, 1)
        user_len_school = user_data[6].float().to(device)    # (b, 1)

        ##### load course data #########
        course_set = course_data[0].long().to(device)
        course_user = course_data[1].long().to(device)
        course_school = course_data[2].long().to(device)
        course_teacher = course_data[3].long().to(device)
        course_len_u = course_data[4].float().to(device)
        course_len_teacher = course_data[5].float().to(device)

        ### calculate user representation###
        user_emb = self.user_embed(user_idx).squeeze()  # (b, d)
        # print('user emb', user_emb.shape)
        seq_emb = self.course_embed(user_sequence)  # (b, seq, d)
        user_teacher_emb = self.teacher_embed(user_teachers)  # (b, t, d)
        user_school_emb = self.school_embed(user_school)  # (b, s, d)

        seq_mean = torch.sum(seq_emb, dim=1) / user_len_seq
        user_teacher_mean = torch.sum(user_teacher_emb, dim=1) / user_len_teacher
        user_school_mean = torch.sum(user_school_emb, dim=1) / user_len_school
        user_repsent = (seq_mean + user_teacher_mean + user_school_mean + user_emb) / 3.0  # (b, d)
        # print('user represent', user_repsent.shape)

        #####calculate course representation ##########
        course_emb = self.course_embed(course_set)  # (b, n, d)
        # print('course emb', course_emb.shape)
        # print('lens u ', course_len_u.shape)

        course_users_emb = self.user_embed(course_user)  # (b, n, u, d)
        course_user_mean = torch.sum(course_users_emb, dim=2) / course_len_u.unsqueeze(2)  # (b, n, d)
        # print('course users emb', course_users_emb.shape)
        # print('course users mean emb', course_user_mean.shape)

        course_school_emb = self.school_embed(course_school)  # (b, n, s, d)
        course_school_mean = torch.sum(course_users_emb, dim=2)  # (b, n, d)
        # print('course school emb', course_school_emb.shape)

        course_teacher_emb = self.teacher_embed(course_teacher)  # (b, n, s, d)
        course_teacher_mean = torch.sum(course_teacher_emb, dim=2) / course_len_teacher.unsqueeze(2)  # (b, n, d)

        course_repsent = (course_user_mean + course_school_mean + course_teacher_mean + course_emb) / 4.0  # (b, n, d)

        out = torch.sum(user_repsent.unsqueeze(1) * course_repsent, dim=2)  # (b, n)
        # print(out.shape)

        return out


class myHIN(nn.Module):
    def __init__(self, num_user, num_course, num_teacher, num_school, num_dim=16,
                 lstm_hidden=16):
        super(myHIN, self).__init__()
        self.num_dim = num_dim

        self.user_embed = nn.Embedding(num_user + 1, num_dim, padding_idx=num_user)
        self.course_embed = nn.Embedding(num_course + 1, num_dim, padding_idx=num_course)
        self.teacher_embed = nn.Embedding(num_teacher + 1, num_dim, padding_idx=num_teacher)
        self.school_embed = nn.Embedding(num_school + 1, num_dim, padding_idx=num_school)

        self.user_att = nn.Sequential(nn.Linear(num_dim, num_dim, bias=True),
                                      nn.Sigmoid(),
                                      nn.Linear(num_dim, 1, bias=False))

        self.user_gnn = nn.Sequential(nn.Linear(num_dim * 2, num_dim, bias=True),
                                      nn.ReLU(),
                                      nn.Linear(num_dim, num_dim, bias=True))
        self.user_neibour_gnn = nn.Linear(num_dim, num_dim)

        self.course_att = nn.Sequential(nn.Linear(num_dim, num_dim, bias=True),
                                        nn.Sigmoid(),
                                        nn.Linear(num_dim, 1, bias=False))

        self.course_gnn = nn.Sequential(nn.Linear(num_dim * 2, num_dim, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(num_dim, num_dim, bias=True))

        self.course_neibour_gnn = nn.Linear(num_dim, num_dim)

        # self.lstm = nn.LSTM(input_size=num_dim, hidden_size=lstm_hidden, batch_first=True, num_layers=1)

        self.transformer_Q = nn.Sequential(nn.Linear(num_dim, num_dim),
                                           nn.ReLU())
        self.transformer_K = nn.Sequential(nn.Linear(num_dim, num_dim),
                                           nn.ReLU())

        self.init_param()
        # self.show_param()

    def init_param(self):
        for param in self.parameters():
            nn.init.normal_(param, 0, 1.0/self.num_dim)
    def show_param(self):
        for name, p in self.named_parameters():
            print(name, p)

    def Transformer(self, x):
        """

        :param x:sequence batch*L*d
        :return: transformer output batch*d

        """
        L = x.shape[1]
        Q = self.transformer_Q(x)
        K = self.transformer_K(x)
        K = K.reshape(-1, self.num_dim, L)

        s = (torch.matmul(Q, K)) / (math.sqrt(self.num_dim))
        for i in range(s.size(1)):
            s[:, i, i] = 1e-32
        # for i in range(s.size(1)):
        #     for j in range(s.size(1)):
        #         if j > i: s[:, i, j] = 0
        su = torch.softmax(s, dim=2)
        a = torch.matmul(su, x)
        m = torch.mean(a, dim=1)

        return m

    def forward(self, user_data, course_data, for_pred=False):
        ##### load user data #########
        user_idx = user_data[0].long().to(device)
        user_sequence = user_data[1].long().to(device)
        user_teachers = user_data[2].long().to(device)
        user_school = user_data[3].long().to(device)
        user_len_seq = user_data[4].float().to(device)  # (b, 1)
        user_len_teacher = user_data[5].float().to(device)  # (b, 1)
        user_len_school = user_data[6].float().to(device)  # (b, 1)

        ##### load course data #########
        course_set = course_data[0].long().to(device)
        course_user = course_data[1].long().to(device)
        course_school = course_data[2].long().to(device)
        course_teacher = course_data[3].long().to(device)
        course_len_u = course_data[4].float().to(device)
        course_len_teacher = course_data[5].float().to(device)

        ### calculate user representation###
        user_emb = self.user_embed(user_idx).squeeze(1)  # (b, d)

        seq_emb = self.course_embed(user_sequence)  # (b, seq, d)
        user_teacher_emb = self.teacher_embed(user_teachers)  # (b, t, d)
        user_school_emb = self.school_embed(user_school)  # (b, s, d)

        # seq_represent = torch.sum(seq_emb, dim=1) / user_len_seq
        # lstm_out, (h, c) = self.lstm(seq_emb)
        # print('lstm out', lstm_out.shape)
        seq_represent = self.Transformer(seq_emb)
        # print('seq out', seq_represent.shape)
        user_teacher_mean = torch.sum(user_teacher_emb, dim=1) / user_len_teacher
        # print(user_teacher_mean.shape)
        user_school_mean = torch.sum(user_school_emb, dim=1) / user_len_school

        user_neighbour_represent = torch.cat([seq_represent.unsqueeze(1), user_teacher_mean.unsqueeze(1), user_school_mean.unsqueeze(1)], dim=1)  # (b,3,d)
        # print('user neighbour', user_neighbour_represent.shape)
        # att_weight = torch.softmax(self.user_att(user_neighbour_represent), dim=1)        # (b, 3)
        # # print('att weight', att_weight.shape)
        # neighbour_agg = torch.sum(user_neighbour_represent * att_weight, dim=1)     # (b, d)
        neighbour_agg = (seq_represent + user_teacher_mean + user_school_mean)/3
        # print('neighbour agg', neighbour_agg.shape)
        # print(user_emb.shape)
        # user_final_represent = torch.cat([user_emb, neighbour_agg], dim=1)                      # (b, 2d)
        user_final_represent = self.user_gnn(torch.cat([user_emb, neighbour_agg], dim=1))


        #####calculate course representation ##########
        course_emb = self.course_embed(course_set)  # (b, n, d)

        course_users_emb = self.user_embed(course_user)  # (b, n, u, d)
        course_user_mean = torch.sum(course_users_emb, dim=2) / course_len_u.unsqueeze(2)  # (b, n, d)

        course_school_emb = self.school_embed(course_school)  # (b, n, s, d)
        course_school_mean = torch.sum(course_school_emb, dim=2)  # (b, n, d)
        # print('course school emb', course_school_emb.shape)

        course_teacher_emb = self.teacher_embed(course_teacher)  # (b, n, s, d)
        course_teacher_mean = torch.sum(course_teacher_emb, dim=2) / course_len_teacher.unsqueeze(2)  # (b, n, d)

        course_neighbour_represent = torch.cat([course_user_mean.unsqueeze(2), course_school_mean.unsqueeze(2), course_teacher_mean.unsqueeze(2)], dim=2)      # (b, n, 3, d)
        # print(course_neighbour_represent.shape)
        course_att_weight = torch.softmax(self.course_att(course_neighbour_represent), dim=2)      # (b, n, 3)
        # print(course_att_weight.shape)
        course_neighbour_agg = torch.sum(course_neighbour_represent * course_att_weight, dim=2)  # (b, n, d)

        # course_final_represent = torch.cat([course_emb, course_neighbour_agg], dim=2)         # (b, n, 2d)
        # course_final_represent = self.course_gnn(torch.cat([course_emb, course_neighbour_agg], dim=2))
        course_final_represent = course_emb
        # print('course final', course_final_represent.shape)


        out = torch.sum(user_final_represent.unsqueeze(1) * course_final_represent, dim=2)  # (b, n)
        # print(out)
        # print(out.shape)

        return out
