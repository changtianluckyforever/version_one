import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from network import Network, DuelNetwork

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, duel=False, double=False, use_icm=False, noisy=False, dpav =False, averaged=False, maxmin_dqn = False, bayesian_dqn = False, num_target_net=4, net_parameter=0):
        super(DQN, self).__init__()
        lr = 0.001
        self.input_size = input_size
        self.duel = duel
        network = DuelNetwork if duel else Network
        self.num_target_net = num_target_net
        self.maxmin_dqn = maxmin_dqn
        if self.maxmin_dqn:
            self.model1 = network(input_size, hidden_size, output_size, noisy)
            self.optimizer1 = optim.RMSprop(self.model1.parameters(), lr=lr)
            self.model2 = network(input_size, hidden_size, output_size, noisy)
            self.optimizer2 = optim.RMSprop(self.model2.parameters(), lr=lr)
            self.model3 = network(input_size, hidden_size, output_size, noisy)
            self.optimizer3 = optim.RMSprop(self.model3.parameters(), lr=lr)
            self.model4 = network(input_size, hidden_size, output_size, noisy)
            self.optimizer4 = optim.RMSprop(self.model4.parameters(), lr=lr)
        else:
            self.model = network(input_size, hidden_size, output_size, noisy)
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)


        # hyper parameters
        self.max_norm = 1e-3
        # lr = 0.001
        self.tau = 1e-2
        self.regc = 1e-3

        self.icm = Network(hidden_size + input_size, hidden_size, input_size)
        self.action_emb = nn.Embedding(input_size, hidden_size)
        self.icm_optim = optim.Adam(self.icm.parameters(), lr=lr)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        # self.optimizer = optim.RMSprop(
        #         self.model.parameters(),
        #         lr=lr)
        self.double = double
        self.use_icm = use_icm
        self.dpav = dpav
        self.net_parameter = net_parameter
        self.averaged = averaged
        self.target_update_period = 3
        self.update_target_net_index = 0
        self.bayesian_dqn = bayesian_dqn

        if self.averaged:
            self.target_model1 = network(input_size, hidden_size, output_size)
            self.target_model1.load_state_dict(self.model.state_dict())
            self.target_model2 = network(input_size, hidden_size, output_size)
            self.target_model2.load_state_dict(self.model.state_dict())
            self.target_model3 = network(input_size, hidden_size, output_size)
            self.target_model3.load_state_dict(self.model.state_dict())
            self.target_model4 = network(input_size, hidden_size, output_size)
            self.target_model4.load_state_dict(self.model.state_dict())

        elif self.maxmin_dqn:
            self.target_model1 = network(input_size, hidden_size, output_size)
            self.target_model1.load_state_dict(self.model1.state_dict())
            self.target_model2 = network(input_size, hidden_size, output_size)
            self.target_model2.load_state_dict(self.model2.state_dict())
            self.target_model3 = network(input_size, hidden_size, output_size)
            self.target_model3.load_state_dict(self.model3.state_dict())
            self.target_model4 = network(input_size, hidden_size, output_size)
            self.target_model4.load_state_dict(self.model4.state_dict())

        else:
            self.target_model = network(input_size, hidden_size, output_size, noisy)
            self.target_model.load_state_dict(self.model.state_dict())

        if use_cuda:
            self.cuda()
            print('we are on GPU now!!!!!!')

    def update_fixed_target_network(self, count_num):
        #self.target_model.load_state_dict(self.model.state_dict())
        # this is original update code
        # for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
        #     target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        # we will move to new code
        if self.averaged:
            target_models = [self.target_model1, self.target_model2, self.target_model3, self.target_model4]  # self.target_model5   self.target_model4
            if count_num % self.target_update_period == 0:
                target_models[self.update_target_net_index].load_state_dict(self.model.state_dict())
                self.update_target_net_index = (self.update_target_net_index + 1) % self.num_target_net

        elif self.maxmin_dqn:
            self.target_model1.load_state_dict(self.model1.state_dict())
            self.target_model2.load_state_dict(self.model2.state_dict())
            self.target_model3.load_state_dict(self.model3.state_dict())
            self.target_model4.load_state_dict(self.model4.state_dict())

        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def Variable(self, x):
        x = x.detach()
        if use_cuda:
            x = x.cuda()
        return x
        #return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    def singleBatch(self, raw_batch, params, discount_val, count_num):

        gamma = params.get('gamma', 0.9)
        
        batch = [np.vstack(b) for b in zip(*raw_batch)]
        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))
        a = self.Variable(torch.LongTensor(batch[1]))
        r = self.Variable(torch.FloatTensor(batch[2]))
        s_prime = self.Variable(torch.FloatTensor(batch[3]))
        done = self.Variable(torch.FloatTensor(np.array(batch[4]).astype(np.float32)))
        i_r = self.Variable(torch.zeros(1)) 
        if self.use_icm:
            s_pred = self.icm(torch.cat([s.detach(), self.action_emb(a.detach()).squeeze()], -1))
            icm_loss = F.mse_loss(s_pred, s_prime.detach(), reduce=False)
            i_r = icm_loss.sum(-1).detach()
            i_r_norm = (i_r - i_r.mean()) / i_r.std()
            r = r + i_r_norm.unsqueeze(1)
            icm_loss = icm_loss.mean()
            self.icm_optim.zero_grad()
            icm_loss.backward()
            self.icm_optim.step()

        if self.maxmin_dqn:
            update_index = np.random.choice(list(range(self.num_target_net)))
            models = [self.model1, self.model2, self.model3, self.model4]
            q = models[update_index](s)
        elif self.bayesian_dqn:
            for_list = []
            for _ in xrange(8):
                q_prime = self.model.bayes(s).detach()
                for_list.append(q_prime.numpy())
            for_list = np.array(for_list)
            for_list = torch.as_tensor(for_list)
            q = torch.mean(for_list, dim=0)

        else:
            q = self.model(s)
        # q = self.model(s)
        if self.double:
            q_prime = self.model(s_prime).detach()
            a_prime = q_prime.max(1)[1]
            q_target_prime = self.target_model(s_prime).detach()
            q_target_prime = q_target_prime.gather(1, a_prime.unsqueeze(1))
            q_target = r + gamma * q_target_prime * (1 - done)

        elif self.averaged:
            print('we are using averaged dqn')
            q_sum = self.target_model1(s_prime).clone().detach()
            q2 = self.target_model2(s_prime).detach()
            q3 = self.target_model3(s_prime).detach()
            q4 = self.target_model4(s_prime).detach()   #if we want to change the number of target nets, change here.
            q_sum = q_sum + q2 + q3 + q4
            q_prime = ( q_sum.max(1)[0].unsqueeze(1)/ self.num_target_net )
            q_target = r + gamma * q_prime * (1 - done)

        elif self.maxmin_dqn:
            print('we are using maxmin dqn')
            targetmodellist = [self.target_model1, self.target_model2, self.target_model3, self.target_model4]
            q_min = self.target_model1(s_prime).clone().detach()
            for i in range(1, self.num_target_net):
                q_m = targetmodellist[i](s_prime).detach()
                q_min = torch.min(q_min, q_m)
            q_prime = q_min.max(1)[0].unsqueeze(1)
            q_target = r + gamma * q_prime * (1 - done)

        elif self.dpav:
            q_prime = self.target_model(s_prime).detach()
            q_prime_max = q_prime.max(1)[0].unsqueeze(1)
            q_prime_min = q_prime.min(1)[0].unsqueeze(1)
            if self.net_parameter:
                discount_val = self.model.calculate_discount_fac(s_prime)
                new_q_prime = discount_val * q_prime_min + (1 - discount_val) * q_prime_max
            else:
                new_q_prime = discount_val * q_prime_min + (1 - discount_val) * q_prime_max
            q_target = r + gamma * new_q_prime * (1 - done)

        elif self.bayesian_dqn:
            blist = []
            for _ in xrange(50):
                q_prime = self.target_model.bayes(s_prime).detach()
                blist.append(q_prime.numpy())
            blist = np.array(blist)
            blist = torch.as_tensor(blist)
            q_prime = torch.mean(blist, dim = 0)
            ws = F.softmax(q_prime, dim=1)
            q_prime = q_prime * ws
            new_q_prime = torch.sum(q_prime, dim = 1, keepdim = True)
            q_target = r + gamma * new_q_prime * (1 - done)

        else:
            q_prime = self.target_model(s_prime).detach()
            q_prime = q_prime.max(1)[0].unsqueeze(1)
            q_target = r + gamma * q_prime * (1 - done)
        q_pred = torch.gather(q, 1, a)
        if self.bayesian_dqn:
            pl = torch.mean(self.model.rate1(s)).item() + 0.000001
            loss = F.mse_loss(q_pred, q_target) + 0.2*torch.norm(self.model.layer1.weight.data)*torch.norm(self.model.layer1.weight.data) - 0.1*self.input_size*(-pl*torch.log(pl)-(1-pl)*torch.log(1-pl))

        else:
            loss = F.mse_loss(q_pred, q_target)
        err = torch.abs(q_pred - q_target).detach()
    

        reg_loss = 0 
        '''
        # L2 regularization
        for name, p in self.model.named_parameters():
            if name.find('weight') != -1:
                reg_loss += self.regc * 0.5 * p.norm(2) / s.size(0)
        '''
        self.update_fixed_target_network(count_num)
        if self.maxmin_dqn:
            optimizers = [self.optimizer1, self.optimizer2, self.optimizer3, self.optimizer4]
            optimizers[update_index].zero_grad()
            (loss + reg_loss).backward()
            clip_grad_norm(models[update_index].parameters(), self.max_norm)
            optimizers[update_index].step()
        else:
            self.optimizer.zero_grad()
            (loss + reg_loss).backward()
            clip_grad_norm(self.model.parameters(), self.max_norm)
            self.optimizer.step()
        # self.optimizer.zero_grad()
        # (loss + reg_loss).backward()
        # clip_grad_norm(self.model.parameters(), self.max_norm)
        # self.optimizer.step()
        # self.model.sample_noise()
        # self.target_model.sample_noise()
        ###########
        ##############

        return {'cost': {'reg_cost': reg_loss, 'loss_cost': loss.item(), 
            'total_cost': (loss + reg_loss).item()}, 'error':err.cpu().numpy(),
            'intrinsic_reward': i_r.mean().cpu().numpy()}

    def get_intrinsic_reward(self, state, next_state, action):
        state = self.Variable(torch.from_numpy(state.astype(np.float32)))
        next_state = self.Variable(torch.from_numpy(next_state.astype(np.float32)))
        action = self.Variable(torch.from_numpy(action.astype(np.int64))).view(1, 1)
        state_pred = self.icm(torch.cat([state, self.action_emb(action).squeeze(0)], -1))
        icm_loss = F.mse_loss(state_pred, next_state.detach(), reduce=False)
        i_r = icm_loss.sum(-1).detach()
        i_r = (i_r - i_r.mean()) / (i_r.std() + 1e-10)
        return i_r.cpu().numpy()


    def predict(self, inputs, a, predict_model, get_q=False):
        inputs = self.Variable(torch.from_numpy(inputs).float())
        if self.maxmin_dqn:
            models = [self.model1, self.model2, self.model3, self.model4]
            q_min = self.model1(inputs, True)
            for i in range(1, self.num_target_net):
                q = models[i](inputs, True)
                q_min = torch.min(q_min, q)
            q, act = torch.max(q_min, 1)

        elif self.bayesian_dqn:
            predict_list = []
            for _ in xrange(8):
                q_prime = self.model.bayes(inputs).detach()
                predict_list.append(q_prime.numpy())
            predict_list = np.array(predict_list)
            predict_list = torch.as_tensor(predict_list)
            q = torch.mean(predict_list, dim=0)
            q, act = torch.max(q, 1)
        else:
            # print( next( self.model.parameters() ).device )
            q, act = torch.max(self.model(inputs, True), 1)
        # q, act = torch.max(self.model(inputs, True), 1)
        act = act.cpu().data.numpy()[0]
        if get_q:
            print("the get_q value is:", get_q)
            return act, q.item()
        return act


    # def save_model(self, model_path):
    #     torch.save(self.model.state_dict(), model_path)
    #     print "model saved."
    #
    # def load_model(self, model_path):
    #     self.model.load_state_dict(torch.load(model_path))
    #     print "model loaded."
