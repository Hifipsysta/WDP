import pickle
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


def open_from_pickle(head_path, save_dir, save_data):
    save_path = head_path + '/' + save_dir + '/' + save_data + '.pickle'
    save_file = open(save_path, 'rb')
    return pickle.load(save_file)


def download_files(head_path):
    Wasser_privacy_list = open_from_pickle(head_path, 'privacy_dir', 'Wasser_privacy_list')
    Renyi_privacy_list = open_from_pickle(head_path, 'privacy_dir', 'Renyi_privacy_list')
    pure_privacy_list = open_from_pickle(head_path, 'privacy_dir', 'pure_privacy_list')
    Wasser_advance_history = open_from_pickle(head_path, 'privacy_dir', 'Wasser_advance_history')
    Bayesian_advance_history = open_from_pickle(head_path, 'privacy_dir', 'Bayesian_advance_history')
    moment_advance_history = open_from_pickle(head_path, 'privacy_dir', 'moment_advance_history')

    train_loss_list = open_from_pickle(head_path, 'evaluation_dir', 'train_loss_list')
    valid_loss_list = open_from_pickle(head_path, 'evaluation_dir', 'valid_loss_list')
    time_consume_history = open_from_pickle(head_path, 'evaluation_dir', 'time_consume_history')
    gradient_norm_history = open_from_pickle(head_path, 'evaluation_dir', 'gradient_norm_history')
    
    return Wasser_privacy_list, Renyi_privacy_list, pure_privacy_list, \
    Wasser_advance_history, Bayesian_advance_history, moment_advance_history,\
    train_loss_list, valid_loss_list, gradient_norm_history, time_consume_history


def log_func(input_data):
    out_data = [np.log(i) for i in input_data]
    return out_data




'''
Wasser_file = open('privacy_dir/Wasser_privacy_list.pickle', 'rb')
Renyi_file = open('privacy_dir/Renyi_privacy_list.pickle', 'rb')
pure_file = open('privacy_dir/pure_privacy_list.pickle', 'rb')
Wasser_adv = open('privacy_dir/Wasser_advance_history.pickle', 'rb')
Bayesian_adv = open('privacy_dir/Bayesian_advance_history.pickle', 'rb')
mement_acc = open('privacy_dir/moment_advance_history.pickle', 'rb')


Wasser_privacy_list = pickle.load(Wasser_file)
Renyi_privacy_list = pickle.load(Renyi_file)
pure_privacy_list = pickle.load(pure_file)
Wasser_advance_history = pickle.load(Wasser_adv)
Bayesian_advance_history = pickle.load(Bayesian_adv)
moment_advance_history = pickle.load(mement_acc)
'''




def plot_privacy(head_path, epoch_list, pure_privacy_list, Renyi_privacy_list, Wasser_privacy_list,
    moment_advance_history, Bayesian_advance_history, Wasser_advance_history):
    plt.figure()
    plt.plot(epoch_list, pure_privacy_list, label='Classical DP')
    plt.plot(epoch_list, Renyi_privacy_list, label = 'Renyi DP')
    plt.plot(epoch_list, Wasser_privacy_list, label = 'Wasserstein DP')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=17)
    plt.ylabel('Epsilon', fontsize=17)
    plt.legend(fontsize=15, loc=2)
    plt.savefig(head_path + '/' +'result_dir/privacy_loss_instant.png', bbox_inches='tight')
    plt.show()


    #def plot_instant():
    plt.figure()
    plt.plot(epoch_list, moment_advance_history, label = 'Moment')
    plt.plot(epoch_list, Bayesian_advance_history, label = 'Bayesian')
    plt.plot(epoch_list, Wasser_advance_history, label = 'Wasserstein')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=17)
    plt.ylabel('$log \epsilon$', fontsize=17)
    plt.ylim(0,26)
    plt.legend(fontsize=15, loc=2)
    plt.savefig(head_path + '/' + 'result_dir/privacy_loss_accountant.png', bbox_inches='tight')
    plt.show()

def plot_grad_norm(head_path, gradient_norm_history):
    #grad_file = open('evaluation_dir/gradient_norm_history.pickle', 'rb')
    #gradient_norm_history = pickle.load(grad_file)

    plt.figure()
    plt.plot(gradient_norm_history)  #bins=np.linspace(0,8,40)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xlabel('Epoch', fontsize=17)
    plt.ylabel('Norms of Gradients', fontsize=17)
    plt.savefig(head_path + '/' +'result_dir/gradient_norm.png', bbox_inches='tight')
    plt.show()



def plot_loss(head_path, train_loss_list, valid_loss_list):
    #train_file = open('evaluation_dir/train_loss_list.pickle', 'rb')
    #valid_file = open('evaluation_dir/valid_loss_list.pickle' , 'rb')
    #train_loss_list = pickle.load(train_file)
    #valid_loss_list = pickle.load(valid_file)
    plt.figure()
    plt.plot(train_loss_list, label = 'train loss', )
    plt.plot(valid_loss_list, label = 'valid loss')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    plt.legend(fontsize=15, loc=2)
    plt.savefig(head_path + '/' +'result_dir/train_valid_loss.png', bbox_inches='tight')


def plot_time(head_path, time_consume_history):
    plt.figure()
    plt.plot(time_consume_history)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=17)
    plt.ylabel('Time consume', fontsize=17)
    plt.legend(fontsize=15)
    plt.show()


def main_plot(head_path):
    Wasser_privacy_list, Renyi_privacy_list, pure_privacy_list, \
    Wasser_advance_history, Bayesian_advance_history, moment_advance_history,\
    train_loss_list, valid_loss_list, gradient_norm_history, \
    time_consume_history = download_files(head_path)

    Wasser_advance_history = log_func(Wasser_advance_history)
    Bayesian_advance_history = log_func(Bayesian_advance_history)
    moment_advance_history = log_func(moment_advance_history)

    n_epcohs = len(Wasser_privacy_list)
    epoch_list = [(i+1) for i in range(n_epcohs)]
    plot_privacy(head_path, epoch_list, pure_privacy_list, Renyi_privacy_list, Wasser_privacy_list,
    moment_advance_history, Bayesian_advance_history, Wasser_advance_history)
    #print('=='*20, gradient_norm_history)
    plot_grad_norm(head_path, gradient_norm_history)
    plot_loss(head_path, train_loss_list, valid_loss_list)
    plot_time(head_path, time_consume_history)


main_plot('MNIST_Clip/Clip=10')

'''
advanced_composition = []
def advanced_comp_alg(epsilon, delta_prime, k):
    right_item = k * epsilon * (np.exp(epsilon) - 1)
    left_item = np.sqrt(2 * k * np.log(1/delta_prime)) * epsilon
    return left_item + right_item

advanced_com = 0
#advanced_comp_alg(epsilon = , delta_prime = 1e-3)
for epoch in range(len(pure_network_privacy_history)):
    advanced_com = advanced_comp_alg(epsilon = pure_network_privacy_history[epoch], delta_prime = 1e-2, k=25)
    advanced_composition.append(advanced_com)
'''

#Renyi_plot_list = [Renyi_privacy_list_concat[0][1][j][-1] for j in range(len(Renyi_privacy_list_concat[0][1]))]
#Wasser_plot_list = [Wasser_privacy_list_concat[0][1][j][-1] for j in range(len(Wasser_privacy_list_concat[0][1]))]

#print(pure_privacy_list_concat)
#print(Wasser_privacy_list_concat[1])
#print(Wasser_privacy_list_concat[0][1][0])
#print(len(Wasser_privacy_list_concat[0][1]))
