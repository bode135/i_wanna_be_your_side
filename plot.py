from args import arg
import numpy as np
import matplotlib.pyplot as plt


# PLOT = arg.Plot

def plot_list(Rewards , title = None, Plot_mean_reward = arg.Plot_mean_reward, cut_len = 10):
    # print mean_ys
    if (Plot_mean_reward):

        # y = np.zeros(len(Rewards))
        y = []
        mean_y = Rewards[0]
        for i in range(len(Rewards)):
            # if(i%100 == 0 and i <= len(Rewards)-100):
            if (i % cut_len == 0):
                end_i = i + cut_len
                if (end_i > len(Rewards) - 1):
                    end_i = len(Rewards) - 1
                    mean_y = np.mean(Rewards[i:end_i])
                y.append(np.mean(Rewards[i:end_i]))
            # y[i] = mean_y
        y
        plt.plot(y)
        plt.show()
        return 1

    # len(Rewards)
    color = 'blue'
    ys = Rewards  # 放大到达终点前的曲线变化
    xs = np.arange(0, len(ys))
    plt.plot(xs, ys, color=color)

    # plot the (mean ± 2*std)
    if (1):
        ys_avg = ys  # 中心线
        ys_std = np.ones(len(ys)) * np.std(ys)  # 方差

        # 最近邻方差
        if (1):
            cut_len = 10

            for i in range(len(ys)):
                if (i > (len(ys) - cut_len // 2)):
                    ys_std[i] = np.std(ys[-cut_len:])
                    continue
                if (i < cut_len // 2):
                    ys_std[i] = np.std(ys[:cut_len])
                else:
                    # i = 10
                    ys_std[i] = np.std(Rewards[i - cut_len // 2:i + cut_len // 2])

        r1 = list(map(lambda x: x[0] - x[1], zip(ys_avg, ys_std)))
        r2 = list(map(lambda x: x[0] + x[1], zip(ys_avg, ys_std)))

        plt.fill_between(xs, r1, r2, color=color, alpha=0.1)

    if(title == None):
        plt.xlabel('episode')
        plt.ylabel('Reward')
    else:
        plt.xlabel(title[0])
        plt.ylabel(title[1])
    plt.show()

    # env.close()

    1



def plot_mean_loss(ys):
        cut_len = 10
        # y = np.zeros(len(ys))
        y = []
        mean_y = ys[0]
        for i in range(len(ys)):
            # if(i%100 == 0 and i <= len(ys)-100):
            if (i % cut_len == 0):
                end_i = i + cut_len
                if (end_i > len(ys) - 1):
                    end_i = len(ys) - 1
                    mean_y = np.mean(ys[i:end_i])
                y.append(np.mean(ys[i:end_i]))
        plt.plot(y)
        plt.xlabel('episode')
        plt.ylabel('Loss')
        plt.show()

def plot_loss(loss_s):
    # if (plot_mean_loss):
    #     plot_mean_loss(loss_s)
    # else:
    color = 'blue'
    ys = loss_s  # 放大到达终点前的曲线变化
    xs = np.arange(0, len(ys))
    plt.plot(xs, ys, color=color)
    plt.xlabel('episode')
    plt.ylabel('Loss')
    plt.show()

def plot_net(net, plot_type = 0 ):
    # net = self.eval_net
    plot_type = ['name', 'data'][plot_type]
    if(plot_type == 'name'):
        for name, parms in net.named_parameters():
            print('---->  name:{:10} ---->  grad_requirs: {} ---->  grad_value: {}'.format(name.rjust(15),
                                                                                           parms.requires_grad, str(
                    list(parms.grad.shape)).rjust(15)))
    if(plot_type == 'data'):
        for name, parms in net.named_parameters():
            print('---->  name:{} ---->  grad_requirs: {} ---->  grad_value: {}'.format(name.rjust(15),
                                                                                           parms.requires_grad, parms.grad ))

if __name__ == '__main__':
    PLOT_MEAN_LOSS = PLOT_LOSS = 1
    rewards = [1,2,3,4,5,6]*100
    plot_list(rewards, False)
    plot_list(rewards, Plot_mean_reward = True, cut_len= 100)

    loss_s = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
    plot_loss(loss_s)
    plot_mean_loss(loss_s)