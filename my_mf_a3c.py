import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from mycode.shared_adam import SharedAdam
from mycode.utils import v_wrap, set_init, push_and_pull, record
import gym
import os
import numpy as np
from mycode.damo import vk
from mycode.my_time import Time
import cv2
from compression import resize_keep_aspectratio
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, 512)
        #self.head = nn.Linear(512, a_dim)
        self.pi1 = nn.Linear(self.fc, 128)
        self.v1 = nn.Linear(self.fc, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v2 = nn.Linear(128, 1)

        #self.pi1 = nn.Linear(s_dim, 128)
        #self.v1 = nn.Linear(s_dim, 128)
        #self.v1 = self.pi1     # can use

        #self.pi2 = nn.Linear(128, a_dim)
        #self.v2 = nn.Linear(128, 1)

        ### set_init(layers)
        def init_params():
            pass
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # batch, channel, w, h
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))



        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

# reference
if(1):

    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
        return out


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)

        1


    class ActorCritic(torch.nn.Module):
        def __init__(self, num_inputs, action_space):
            super(ActorCritic, self).__init__()
            #nn.Conv2d(screen, 32, 3, stride=2, padding=1)

            self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

            self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

            num_outputs = action_space.n
            self.critic_linear = nn.Linear(256, 1)
            self.actor_linear = nn.Linear(256, num_outputs)

            self.apply(weights_init)
            self.actor_linear.weight.data = normalized_columns_initializer(
                self.actor_linear.weight.data, 0.01)
            self.actor_linear.bias.data.fill_(0)
            self.critic_linear.weight.data = normalized_columns_initializer(
                self.critic_linear.weight.data, 1.0)
            self.critic_linear.bias.data.fill_(0)

            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

            self.train()

        def forward(self, inputs):
            inputs, (hx, cx) = inputs
            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx

            return self.critic_linear(x), self.actor_linear(x), (hx, cx)
    1


    class DQN(nn.Module):

        def __init__(self, h, w, outputs):
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=5, stride=2):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))
    1

# functions
if(1):
    def cv_img(image):  # cv plot image
        cv2.imshow('image', image)
        print(cv2.waitKey(0))
        cv2.destroyAllWindows()
    def plt_img(image):  # plt plot image
        plt.imshow(image, cmap='gray')
        plt.show()

    1

env.reset()

def get_screen():
    # get_screen
    screen_0 = env.render(mode='rgb_array')
    screen = screen_0.transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape



    # process screen
    screen_0.shape
    image_src = screen_0

    dst_size = (100, 100)

    image = resize_keep_aspectratio(image_src, dst_size)

    image = cv2.cvtColor(image_src, cv2.COLOR_RGB2GRAY)
    image.shape

    plt_img(image)

    # env.close()
    pass
if(0):
    env.reset()
    env.render()
    # env.close()
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    c,h, w = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = screen.unsqueeze(0)
    screen_np = screen.cpu().squeeze(0).permute(1, 2, 0).numpy()
    screen.shape
    screen_np.shape
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(screen_np)
    # plt.show()

    dqn = DQN(h, w, 2) # Height * Width , n_action

    tt = Time()
    for i in range(100):
        dqn.forward(screen)
    print(tt.now())

    screen.shape
    xx = torch.unsqueeze(screen, 0)
    torch.cat([screen, screen],-1).shape

    l_screen  = []
    for i in range(100):
        l_screen.append(screen)
    end = []
    for screen0 in l_screen:
        screen0 = screen0.unsqueeze(0)
        try:
            end = torch.cat([end, screen0])
        except:
            end = screen0
        print(end.shape)
    dqn.forward(end)







    screen_batch = torch.cat([xx, xx])
    screen_batch.shape
    pred_a = dqn.forward(screen_batch)
    a = torch.LongTensor([[1],[1]])
    a.shape
    q_values = dqn(screen_batch).gather(1, a)
    q_values

    env.close()

    xx = torch.rand(10)
    torch.cat([xx, xx])


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if(0):
    if __name__ == "__main__":

        gnet = Net(N_S, N_A)  # global network
        gnet.share_memory()  # share the global parameters in multiprocessing
        opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
        # opt = torch.optim.Adam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))

        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

        # parallel training
        workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]

        if (0):
            opt.state_dict()
            opt.param_groups
            for i in range(1000):    print('---------------------')
            p = opt.param_groups[0]['params'][0]
            opt.state[p]

        [w.start() for w in workers]
        res = []  # record episode reward to plot
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]

        import matplotlib.pyplot as plt

        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()