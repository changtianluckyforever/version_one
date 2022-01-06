'''
created on Mar 08, 2018
@author: Shang-Yu Su
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

from collections import OrderedDict

use_cuda = torch.cuda.is_available()


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

        self.discount_fac = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(input_size, hidden_size)),
            ('Relu', nn.ReLU()),
            ('l2', nn.Linear(hidden_size, 1)),
            ('sig', nn.Sigmoid())]))



    def forward(self, inputs, testing=False):
        return self.model(inputs)

    def calculate_discount_fac(self, next_states):
        return self.discount_fac(next_states)


class DuelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelNetwork, self).__init__()
        self.advantage = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, output_size))
        self.value_func = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, inputs, testing=False):
        v = self.value_func(inputs)
        adv = self.advantage(inputs)
        return v.expand(adv.size()) + adv - adv.mean(-1).unsqueeze(1).expand(adv.size())


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, duel=False, double=False, pav=False, averaged_dqn=False, num_target_net=1, net_parameter = 0, maxmin_dqn = False, sunrise=False, b_size = 16):
        super(DQN, self).__init__()

        network = DuelNetwork if duel else Network

        # model
        # self.model = network(input_size, hidden_size, output_size)

        # target model
        # self.target_model = network(input_size, hidden_size, output_size)
        # first sync
        # self.target_model.load_state_dict(self.model.state_dict())

        # hyper parameters
        self.num_target_net = num_target_net
        self.update_target_net_index = 0
        self.net_parameter = net_parameter
        self.gamma = 0.9
        self.reg_l2 = 1e-3
        self.max_norm = 1
        self.target_update_period = 3
        lr = 0.001
        self.averaged_dqn = averaged_dqn
        self.double = double
        self.pav = pav
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.batch_count = 0

        # model
        self.model = network(input_size, hidden_size, output_size)

        if self.averaged_dqn:
            # self.target_model = [None] * self.num_target_net
            # for i in range(self.num_target_net):
            #     self.target_model[i] = network(input_size, hidden_size, output_size)
            #     self.target_model[i].load_state_dict(self.model.state_dict())
            #     # self.target_model[i].eval()
            self.target_model1 = network(input_size, hidden_size, output_size)
            self.target_model1.load_state_dict(self.model.state_dict())
            self.target_model2 = network(input_size, hidden_size, output_size)
            self.target_model2.load_state_dict(self.model.state_dict())
            self.target_model3 = network(input_size, hidden_size, output_size)
            self.target_model3.load_state_dict(self.model.state_dict())
            self.target_model4 = network(input_size, hidden_size, output_size)
            self.target_model4.load_state_dict(self.model.state_dict())

        else:
            # target model
            self.target_model = network(input_size, hidden_size, output_size)
            # first sync
            self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

        # target model
        # self.target_model = network(input_size, hidden_size, output_size)
        # first sync
        # self.target_model.load_state_dict(self.model.state_dict())
        # self.to(device)
        if use_cuda:
            self.cuda()

    def update_fixed_target_network(self, episode):

        if self.averaged_dqn:
            target_models = [self.target_model1, self.target_model2, self.target_model3, self.target_model4]  # self.target_model5   self.target_model4
            if episode % self.target_update_period == 0:
                target_models[self.update_target_net_index].load_state_dict(self.model.state_dict())
                self.update_target_net_index = (self.update_target_net_index + 1) % self.num_target_net
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    # def Variable(self, x):
    #     return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    def Variable(self, x):
        x = x.detach()
        if use_cuda:
            x = x.cuda()
        return x

    def singleBatch(self, batch, discount_factor):
        self.optimizer.zero_grad()
        loss = 0

        # each example in a batch: [s, a, r, s_prime, term]
        # s = Variable(torch.FloatTensor(batch[0]), requires_grad=True)  # original requires grad true
        s = self.Variable(torch.FloatTensor(batch[0]))
        a = self.Variable(torch.LongTensor(batch[1]))
        r = self.Variable(torch.FloatTensor([batch[2]]))
        s_prime = self.Variable(torch.FloatTensor(batch[3]))
        term = self.Variable(torch.FloatTensor(batch[4].astype(np.float32)))
        factor = discount_factor

        q = self.model(s)
        # print('have a look')
        # print(q.requires_grad)
        # print('have a look end')
        # q_prime = self.target_model(s_prime)

        if self.double:
            q_prime = self.model(s_prime).detach()
            a_prime = q_prime.max(1)[1]
            q_target_prime = self.target_model(s_prime).detach()
            q_target_prime = q_target_prime.gather(1, a_prime.unsqueeze(1))
            q_target = r + self.gamma * q_target_prime * (1 - term)
        elif self.pav:
            if self.averaged_dqn:
                q_sum = self.target_model1(s_prime).clone().detach()
                q2 = self.target_model2(s_prime).detach()
                q3 = self.target_model4(s_prime).detach()
                q4 = self.target_model5(s_prime).detach()
                q_sum = q_sum + q2 + q3 + q4
                q_prime_max = (q_sum.max(1)[0].unsqueeze(1) / self.num_target_net)
                q_prime_min = (q_sum.min(1)[0].unsqueeze(1) / self.num_target_net)
                new_q_prime = factor * q_prime_min + (1 - factor) * q_prime_max
                q_target = r + self.gamma * new_q_prime * (1 - term)

            else:
                q_prime = self.target_model(s_prime).detach()
                q_prime_max = q_prime.max(1)[0].unsqueeze(1)
                q_prime_min = q_prime.min(1)[0].unsqueeze(1)

                if self.net_parameter:
                    factor = self.model.calculate_discount_fac(s_prime)
                    new_q_prime = factor * q_prime_min + (1 - factor) * q_prime_max
                else:
                    new_q_prime = factor * q_prime_min + (1 - factor) * q_prime_max


                # new_q_prime = factor * q_prime_min + (1 - factor) * q_prime_max
                q_target = r + self.gamma * new_q_prime * (1 - term)
                print('type of factor:', type(factor))

            # q_prime = self.target_model(s_prime).detach()
            # q_prime_max= q_prime.max(1)[0].unsqueeze(1)
            # q_prime_min= q_prime.min(1)[0].unsqueeze(1)
            # new_q_prime = factor*q_prime_min + (1-factor)*q_prime_max
            # q_target= r+self.gamma* new_q_prime*(1-term)
            # print('type of factor:', type(factor))

        elif self.averaged_dqn:
            q_sum = self.target_model1(s_prime).clone().detach()

            q2 = self.target_model2(s_prime).detach()
            q3 = self.target_model3(s_prime).detach()
            q4 = self.target_model4(s_prime).detach()
            q_sum = q_sum + q2 + q3 + q4
            q_prime = (q_sum.max(1)[0].unsqueeze(1) / self.num_target_net)
            q_target = r + self.gamma * q_prime * (1 - term)

        else:
            q_prime = self.target_model(s_prime).detach()
            q_prime = q_prime.max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_prime * (1 - term)
        # print('have a look')
        # print(q.requires_grad)   
        q_pre = torch.gather(q, 1, a)
        # print(q_target.requires_grad)
        # print(q_pre.requires_grad)
        # print('have a look end')

        td_error = q_target - q_pre

        # the batch style of (td_error = r + self.gamma * torch.max(q_prime) - q[a])
        # td_error = r.squeeze_(0) + torch.mul(torch.max(q_prime, 1)[0], self.gamma).unsqueeze(1) * (1-term) - torch.gather(q, 1, a)
        loss += td_error.pow(2).sum()

        # loss.requires_grad = True
        loss.backward()
        clip_grad_norm(self.model.parameters(), self.max_norm)
        self.optimizer.step()

    def predict(self, inputs, get_q = False, eval = False):
        inputs = self.Variable(torch.from_numpy(inputs).float())
        return self.model(inputs, True).cpu().data.numpy()[0]

    def predict_new(self, inputs, a, predict_model, get_q = False, eval = False):
        inputs = self.Variable(torch.from_numpy(inputs).float())
        q, act = torch.max(self.model(inputs, True), 1)
        act = act.cpu().data.numpy()[0]
        if get_q:
            # print("the get_q value is:", get_q)
            return act, q.item()
        return act


    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print "model saved."

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print "model loaded."
