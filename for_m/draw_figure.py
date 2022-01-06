import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="darkgrid")
sns.set(font_scale=1.6)

width = 8
height = 5.8
plt.figure(figsize=(width, height))
linewidth=1.1

def read_performance(path, attribute):
    success_rate = []
    data = json.load(open(path, 'rb'))
    for key in sorted(data[attribute].keys(), key=lambda k: int(k)):
        if int(key) > -1:
            if attribute == 'discriminator_loss':
                success_rate.append(data[attribute][key]/50)
            else:
                success_rate.append(data[attribute][key])

    smooth_num = 1
    d = [success_rate[i*smooth_num:i*smooth_num + smooth_num] for i in xrange(len(success_rate)/smooth_num)]

    success_rate_new = []
    cache = 0
    alpha = 0.85
    for i in d:
        cur = sum(i)/float(smooth_num)
        cache = cache * alpha + (1 - alpha) * cur
        success_rate_new.append(cache)

    return success_rate_new


def show_model_performance(path, record_list=range(1, 4)):
    attributes = ['success_rate', 'ave_reward', 'ave_turns']
    records = {
        'success_rate': {'100': 0, '200': 0, '300': 0},
        'ave_reward': {'100': 0, '200': 0, '300': 0},
        'ave_turns': {'100': 0, '200': 0, '300': 0}
    }
    for i in record_list:
        data = json.load(open('{}_{}/agt_9_performance_records.json'.format(path, i), 'rb'))
        for attribute in records.keys():
            records[attribute]['100'] += data[attribute]['100'] / len(record_list)
            records[attribute]['200'] += data[attribute]['200'] / len(record_list)
            records[attribute]['300'] += data[attribute]['300'] / len(record_list)
    print path
    print records
    return records


def draw(color, marker, linestyle, record_list=range(1,4), model_path="", attribute="success_rate"):
    datapoints = []
    for i in record_list:
        datapoints.append(
            read_performance('{}_{}/agt_9_performance_records.json'.format(model_path, i), attribute))

    min_len = min(len(i) for i in datapoints)
    data = np.asarray([i[0:min_len] for i in datapoints])
    mean = np.mean(data,axis=0)
    var = np.std(data,axis=0)
    l, = plt.plot(range(mean.shape[0]), mean, color, marker=marker, markevery=10, markersize=5, linewidth=linewidth)
    plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=color, alpha=0.2)
    return l


def main(params):
    colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#FF8800', 'purple', 'maroon']
    markers = [',', 'o', '^', 's', 'p', 'd', '1', 'h']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-.', '--', '-', ':']
    global_idx = 1500

    # example
    model_path_list = [
        './deep_dialog/checkpoints/avgdqn_4_1',
        #'./deep_dialog/for_plot/avgdqn_3_1',
        #'./deep_dialog/for_plot/avgdqn_0_1',
        #'./deep_dialog/for_plot/ndueldqn_0_1',
        #'./deep_dialog/for_plot/pavdueldqn_0_1',
        #'./deep_dialog/for_plot/dqn_0_1',
        #'./deep_dialog/for_plot/duelavgdqn_0_1',
        './deep_dialog/checkpoints/avgdqn_3_1',
        './deep_dialog/checkpoints/avgdqn_2_1'

    ]
    #label_list = ['dpav', 'double DQN', 'avg DQN', 'duel DQN', 'duel DPAV', 'DQN', 'duel avg DQN', 'DPAV avg DQN']
    label_list = ['targetnet4', 'targetnet3', 'targetnet2']
    curve_list = []
    for i, model in enumerate(model_path_list):
        record_list = range(1, 4)
        curve_list.append(draw(model_path=model, color=colors[i], marker=markers[i], linestyle=linestyles[i], record_list=record_list))

    plt.grid(True)
    plt.ylabel('Success rate', fontsize = 'xx-small')
    plt.xlabel('Epoch', fontsize = 'xx-small')
    plt.xticks(fontsize='xx-small')
    plt.yticks(fontsize='xx-small')
    plt.legend(curve_list, label_list, loc=4,fontsize = 'xx-small')
    plt.xlim([0, 490])
    plt.ylim([0, 0.9])
    plt.savefig('./figure.pdf', format='pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', dest='result_file', type=str, default='agt_10_performance_records.json', help='path to the result file')

    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)

    main(params)
