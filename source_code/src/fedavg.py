import os
from sklearn import cluster
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
import time
from sklearn.cluster import KMeans
from models import *
from options import args_parser
from utils import average_weights, exp_details, get_datasets, adj_matrix_converter_flatten
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from test import test_img

from sampling import DatasetSplit


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')


    args = args_parser()
    exp_details(args)

    device = args.device

    # load dataset and user groups
    train_dataset, test_dataset, dict_users_train, dict_users_test = get_datasets(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'svhn':
            global_model = CNNSVHN(args=args).to(device)
    # if args.model == 'vgg' and args.arch == 'vgg19':
    #     if args.dataset == 'cifar':
    #         global_model = models.__dict__[args.arch]()
    elif args.model == 'vgg':
        if args.dataset == 'cifar':
            global_model = vgg(dataset = args.dataset).to(device)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    avg_local_test_losses_list, avg_local_test_accuracy_list = [],[]
    avg_local_train_losses_list = []
    print_every = 1

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_train_losses = [], []
        local_test_losses, local_test_accuracy = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:

            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs = dict_users_train[idx])
            w, loss = local_model.update_weights(model = copy.deepcopy(global_model), global_round=epoch)
            trained_local_model = copy.deepcopy(global_model)
            trained_local_model.load_state_dict(w)

            if args.customize_test:

            ################## each client has each test #################################
                test_loader_for_each_client = torch.utils.data.DataLoader(
                    dataset=DatasetSplit(train_dataset, dict_users_test[idx]),
                    shuffle=True,
                )
                test_acc, test_loss = test_img(trained_local_model, test_loader_for_each_client, args)
            ####################################################################################
            else:
                test_loader_for_share = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                shuffle=True   
                )
                test_acc, test_loss = test_img(trained_local_model, test_loader_for_share, args)



            
            local_weights.append(copy.deepcopy(w))
            local_train_losses.append(copy.deepcopy(loss))
            local_test_losses.append(test_loss)
            local_test_accuracy.append(test_acc)

        loss_avg_train_loss = sum(local_train_losses) / len(local_train_losses)
        avg_local_train_losses_list.append(loss_avg_train_loss)

        loss_avg_test_loss = sum(local_test_losses) / len(local_test_losses)
        avg_local_test_losses_list.append(loss_avg_test_loss)
        loss_avg_test_accuracy = sum(local_test_accuracy) / len(local_test_accuracy)
        avg_local_test_accuracy_list.append(loss_avg_test_accuracy)
        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Stats after {epoch+1} global rounds:')
            print(f'Local Avg Training Loss : {loss_avg_train_loss}')
            print(f'Local Avg Test Loss : {loss_avg_test_loss}')
            print(f'Local Avg Test Accuracy : {loss_avg_test_accuracy}')

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

    save_path = '../11_21/fedavg_save/fedavg_{}_{}_{}_iid[{}]_E[{}]_mp_{}_cus_test_{}_cluster_{}_frac_{}_shards_{}/'.format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.message_passing_num, args.customize_test, args.cluster_num, args.frac, args.num_shards)
# Check whether the specified path exists or not
    isExist = os.path.exists(save_path)

    if not isExist:
  # Create a new directory because it does not exist 
        os.makedirs(save_path)
        print("The new directory is created!")
    with open(save_path + 'local_avg_train_loss.txt', 'w') as filehandle:
        for listitem in avg_local_train_losses_list:
            filehandle.write('%s\n' % listitem)

    with open(save_path + 'local_avg_test_losses_list.txt', 'w') as filehandle:
        for listitem in avg_local_test_losses_list:
            filehandle.write('%s\n' % listitem) 

    with open(save_path + 'local_avg_test_accuracy_list.txt', 'w') as filehandle:
        for listitem in avg_local_test_accuracy_list:
            filehandle.write('%s\n' % listitem) 
    
    matplotlib.use('Agg')

        # Plot Loss curve
    plt.figure()
    plt.title('Local Average Training Loss vs Communication rounds')
    plt.plot(range(len(avg_local_train_losses_list)), avg_local_train_losses_list, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_customized_test_{}_train_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs, args.customize_test))

    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Local Average Test Loss vs Communication rounds')
    plt.plot(range(len(avg_local_test_losses_list)), avg_local_test_losses_list, color='k')
    plt.ylabel('Test Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_customized_test_{}_test_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs, args.customize_test))

    plt.figure()
    plt.title('Local Average Test Accuracy vs Communication rounds')
    plt.plot(range(len(avg_local_test_accuracy_list)), avg_local_test_accuracy_list, color='r')
    plt.ylabel('Test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_customized_test_{}_test_accuracy.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs, args.customize_test))
#         avg_local_train_loss.append(loss_avg)

#         # Calculate avg training accuracy over all users at every epoch
#         list_acc, list_loss = [], []
#         global_model.eval()
#         for c in range(args.num_users):
#             local_model = LocalUpdate(args=args, dataset=train_dataset,
#                                       idxs=dict_users_train[idx])
#             acc, loss = local_model.inference(model=global_model)
#             list_acc.append(acc)
#             list_loss.append(loss)
#         train_accuracy.append(sum(list_acc)/len(list_acc))

#         # print global training loss after every 'i' rounds
#         if (epoch+1) % print_every == 0:
#             print(f' \nAvg Training Stats after {epoch+1} global rounds:')
#             print(f'Training Loss : {np.mean(np.array(train_loss))}')
#             print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

#     # Test inference after completion of training
#     test_acc, test_loss = test_inference(args, global_model, test_dataset)

#     print(f' \n Results after {args.epochs} global rounds of training:')
#     print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
#     print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

#     # Saving the objects train_loss and train_accuracy:
#     file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
#         format(args.dataset, args.model, args.epochs, args.frac, args.iid,
#                args.local_ep, args.local_bs)

#     with open(file_name, 'wb') as f:
#         pickle.dump([train_loss, train_accuracy], f)

#     print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

# # PLOTTING (optional)
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')

# # Plot Loss curve
# plt.figure()
# plt.title('Training Loss vs Communication rounds')
# plt.plot(range(len(train_loss)), train_loss, color='r')
# plt.ylabel('Training loss')
# plt.xlabel('Communication Rounds')
# plt.savefig('../fedavg_save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
#             format(args.dataset, args.model, args.epochs, args.frac,
#                     args.iid, args.local_ep, args.local_bs))

# # Plot Average Accuracy vs Communication rounds
# plt.figure()
# plt.title('Average Accuracy vs Communication rounds')
# plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
# plt.ylabel('Average Accuracy')
# plt.xlabel('Communication Rounds')
# plt.savefig('../fedavg_save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
#             format(args.dataset, args.model, args.epochs, args.frac,
#                     args.iid, args.local_ep, args.local_bs))
