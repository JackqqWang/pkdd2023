
# no meassage passing
import seaborn as sns
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
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNSVHN
from test import test_img, test_inference

from sampling import DatasetSplit


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

def list_to_dict(l):
    d = {}
    for i, val in enumerate(l):
        d.setdefault(val, []).append(i)

    return d

if __name__ == '__main__':
    start_time = time.time()
    path_project = os.path.abspath('..')


    args = args_parser()
    exp_details(args)
    device = args.device

    # load dataset
    train_dataset, test_dataset, dict_users_train, dict_users_test = get_datasets(args) 
    # this test_dataset is the typical test part of the benchmark dataset


    # load model
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args).to(device)
        # elif args.dataset == 'fmnist':
        #     global_model = CNNFashion_Mnist(args=args).to(device)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args).to(device)
        elif args.dataset == 'svhn':
            global_model = CNNSVHN(args=args).to(device)

    elif args.model == 'vgg':
        if args.dataset == 'cifar':
            global_model = vgg(dataset = args.dataset).to(device)
    else:
        exit("error: unrecognized model")
    
    global_model.to(device)
    global_model.train()
    print("check global model:")
    print(global_model)


    global_weights = global_model.state_dict()

    #training


    local_avg_train_losses_list, local_avg_train_accuracy = [],[]
    local_avg_test_losses_list, local_avg_test_accuracy_list = [],[]
    val_acc_list, net_list = [], []
    global_test_acc = []
    print_every = 1
    ever_selected_client_idx_at_server = []
    ad_weight_original_matrix_list = []


    for epoch in tqdm(range(args.epochs)): # communication
        local_weights, local_losses, local_model_list= [], [], []

        local_test_losses, local_test_accuracy = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # ever_selected_client_idx_at_server = ever_selected_client_idx_at_server + list(idxs_users)

        idx_user_id = dict(zip([i for i in range(len(idxs_users))], list(idxs_users))) # key is the index, value is the user id

        for idx in idxs_users:
            test_loader_for_each_client = torch.utils.data.DataLoader(
                dataset=DatasetSplit(train_dataset, dict_users_test[idx]),  #labeled data for user i
                shuffle=True,
            )
            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs = dict_users_train[idx])
            if args.customize_test:
                if epoch == 0:
                    w, loss = local_model.update_weights(model = copy.deepcopy(global_model), global_round=epoch)
                    trained_local_model = copy.deepcopy(global_model)
                    trained_local_model.load_state_dict(w)
                    test_acc, test_loss = test_img(trained_local_model, test_loader_for_each_client, args)
                   
                else:
                    if idx in (previous_user_list):
                        for key, value in cluster_client_dict.items():
                            if idx in value:
                                temp_model = copy.deepcopy(global_model)
                                temp_model.load_state_dict(cluster_client_agg_model_dict[key])
                                w, loss= local_model.update_weights(model = temp_model, global_round=epoch) # todo cluster
                                trained_local_model = copy.deepcopy(global_model)
                                trained_local_model.load_state_dict(w)
                                test_acc, test_loss = test_img(trained_local_model, test_loader_for_each_client, args)
                                # test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)
                                
                    else:
                        temp_model = copy.deepcopy(global_model)
                        temp_model.load_state_dict(cluster_merge_one_global_model)
                        w, loss = local_model.update_weights(model = temp_model, global_round=epoch) # todo agg
                        trained_local_model = copy.deepcopy(global_model)
                        trained_local_model.load_state_dict(w)
                        test_acc, test_loss = test_img(trained_local_model, test_loader_for_each_client, args)
                        # test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_each_client)
            else:
                test_loader_for_share = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                shuffle=False
                )
                if epoch == 0:
                    w, loss = local_model.update_weights(model = copy.deepcopy(global_model), global_round=epoch)
                    trained_local_model = copy.deepcopy(global_model)
                    trained_local_model.load_state_dict(w)
                    test_acc, test_loss = test_img(trained_local_model, test_loader_for_share, args)
                    # test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_share)
                    # previous_user_list = idxs_users
                else:
                    if idx in (previous_user_list):
                        for key, value in cluster_client_dict.items():
                            if idx in value:
                                temp_model = copy.deepcopy(global_model)
                                temp_model.load_state_dict(cluster_client_agg_model_dict[key])
                                w, loss= local_model.update_weights(model = temp_model, global_round=epoch) # todo cluster
                                trained_local_model = copy.deepcopy(global_model)
                                trained_local_model.load_state_dict(w)
                                test_acc, test_loss = test_img(trained_local_model, test_loader_for_share, args)
                                # test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_share)
                                
                    else:
                        temp_model = copy.deepcopy(global_model)
                        temp_model.load_state_dict(cluster_merge_one_global_model)
                        w, loss = local_model.update_weights(model = temp_model, global_round=epoch) # todo agg
                        trained_local_model = copy.deepcopy(global_model)
                        trained_local_model.load_state_dict(w)
                        test_acc, test_loss = test_img(trained_local_model, test_loader_for_share, args)
                        # test_acc, test_loss = test_inference(args, trained_local_model, test_loader_for_share)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_test_losses.append(test_loss) # len = user number 
            local_test_accuracy.append(test_acc)
            temp_temp_model = copy.deepcopy(global_model)
            temp_temp_model.load_state_dict(w)
            local_model_list.append(temp_temp_model)
        
        previous_user_list = idxs_users



        loss_avg = sum(local_losses) / len(local_losses)
        local_avg_train_losses_list.append(loss_avg) # avg local client loss

        loss_avg_test_loss = sum(local_test_losses) / len(local_test_losses)
        local_avg_test_losses_list.append(loss_avg_test_loss)
        loss_avg_test_accuracy = sum(local_test_accuracy) / len(local_test_accuracy)
        local_avg_test_accuracy_list.append(loss_avg_test_accuracy)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Stats after {epoch+1} global rounds:')
            print(f'Local Avg Training Loss : {loss_avg}')
            print(f'Local Avg Test Loss : {loss_avg_test_loss}')
            print(f'Local Avg Test Accuracy : {loss_avg_test_accuracy}')
    
        

        local_model_flatten_list = [] # model flatten list
        for each_local_model in local_model_list:
            temp_list = []
            for param in each_local_model.parameters():
                temp_list.extend(torch.flatten(param).cpu().detach().numpy())
            local_model_flatten_list.append(temp_list)
        local_model_flatten_array  = np.array(local_model_flatten_list)
        kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(local_model_flatten_array)

        cluster_client_dict = list_to_dict(list(kmeans.labels_)) # key is the cluster index, value is a list of client index, where client index in each cluster
        
        # 



        cluster_client_model_dict = {}
        for key, value in cluster_client_dict.items():
            cluster_client_model_dict[key] = [local_weights[item] for item in value]  
            # key is the cluster index, value is a list of model parameters, where client index in each cluster
        
        cluster_client_agg_model_dict = {}
        for key,value in cluster_client_model_dict.items():
            cluster_client_agg_model_dict[key] = average_weights(value) # key is the cluster index, value is the aggregate model

        # before message passing, see how the performance is for each aggregation model
        
        global_model_init = [copy.deepcopy(global_model.to(device)) for _ in range(args.cluster_num)]  #[global_model,global_model,global_model ... ] number of cluster

        for key in range(args.cluster_num):
            global_model_init[key].load_state_dict(cluster_client_agg_model_dict[key])


        # ************************************graph building************************************#

        # now we have a list, the length is the cluster number, and each item is the aggregation model for each cluster

        # ad_matrix = adj_matrix_converter_flatten(global_model_init)
         
        # cluster number * cluster number size, binary matrix, 1>= average, 0 < average

        # ************************************message passing ************************************#

        # conduct message passing
        cluster_model_dict_list = [item.state_dict() for item in global_model_init]
        ad_weight_original_matrix, ad_matrix = adj_matrix_converter_flatten(global_model_init)
        if args.message_passing_num:
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    layer_model = []
                    for cluster_model in global_model_init:
                        layer_model.append(torch.flatten(cluster_model.state_dict()[key]).cpu().detach().numpy())
                    layer_para_num = len(torch.flatten(cluster_model.state_dict()[key]).cpu().detach().numpy())
                    model_number = len(global_model_init)
                    empty_matrix = np.zeros((layer_para_num, model_number))
                    for i in range(model_number):
                        empty_matrix[:, i] = layer_model[i]
                    if args.message_passing_num == 1:
                        layer_update = np.matmul(empty_matrix, ad_matrix) 
                    else:
                        layer_update = np.matmul(empty_matrix, ad_matrix) 
                        for _ in range (args.message_passing_num-1):
                            layer_update = np.matmul(layer_update, ad_matrix) 
                        
                    for i in range(model_number):
                        reshape_para = torch.reshape(torch.tensor(layer_update[:,i]), cluster_model.state_dict()[key].shape)
                        cluster_model_dict_list[i][key] = reshape_para
        for cluster_model, state_dict in zip(global_model_init,cluster_model_dict_list):
            cluster_model.load_state_dict(state_dict)
        
        # after message passing, we get update  "global_model_init"

        cluster_merge_one_global_model = average_weights(cluster_model_dict_list)
        ad_weight_original_matrix_list.append(ad_weight_original_matrix)

        # we get new cluster_client_agg_model_dict for next communication round use
        cluster_client_agg_model_dict = {}
        for key in range(len(global_model_init)):
            cluster_client_agg_model_dict[key] = global_model_init[key].state_dict()

    save_path = '../AS2_12_27_exp/fed_{}_{}_{}_iid[{}]_E[{}]_mp_{}_cus_test_{}_cluster_{}_frac_{}_user_num_{}/'.format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.message_passing_num, args.customize_test, args.cluster_num, args.frac, args.num_users)
# Check whether the specified path exists or not
    isExist = os.path.exists(save_path)

    if not isExist:
  
  # Create a new directory because it does not exist 
        os.makedirs(save_path)
        print("The new directory is created!")
    np.save(save_path +'ad_weight_original_matrix_list.npy'.format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.message_passing_num, args.customize_test, args.cluster_num, args.frac), ad_weight_original_matrix_list, allow_pickle=True)
   

    fig = sns.heatmap(ad_weight_original_matrix_list[0], cmap='hot')
    figure = fig.get_figure() 
    figure.savefig(save_path + 'fig_{}'.format(0), dpi=400)
    for i in range(1, len(ad_weight_original_matrix_list)):
        fig = sns.heatmap(ad_weight_original_matrix_list[i], cmap='hot', cbar = False)
        figure = fig.get_figure() 
        figure.savefig(save_path + 'fig_{}'.format(i), dpi=400)
        # global_weights = average_weights(local_weights)
        # global_model.load_state_dict(global_weights)
        # loss_avg = sum(local_losses) / len(local_losses)
        # local_avg_train_loss.append(loss_avg) # avg local client loss

        # loss_avg_test_loss = sum(local_test_losses) / len(local_test_losses)
        # local_test_losses_list.append(loss_avg_test_loss)
        # loss_avg_test_accuracy = sum(local_test_accuracy) / len(local_test_accuracy)
        # local_test_accuracy_list.append(loss_avg_test_accuracy)

        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Stats after {epoch+1} global rounds:')
        #     print(f'Local Avg Training Loss : {loss_avg}')
        #     print(f'Local Avg Test Loss : {loss_avg_test_loss}')
        #     print(f'Local Avg Test Accuracy : {loss_avg_test_accuracy}')
        # test_acc, test_loss = test_inference(args, global_model,test_dataset)
        # global_test_acc.append(test_acc)
    # test_acc, test_loss = test_inference(args, global_model,test_dataset)
    # print(f' \n Results after {args.epochs} global rounds of communication:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Global Test Accuracy: {:.2f}%".format(100*test_acc))
    #
    # local model training
    # local model test
##########################################################################optional#########################################################
    #  update global model weights
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    
    with open(save_path + 'local_avg_train_loss.txt', 'w') as filehandle:
        for listitem in local_avg_train_losses_list:
            filehandle.write('%s\n' % listitem)

    with open(save_path + 'local_avg_test_losses_list.txt', 'w') as filehandle:
        for listitem in local_avg_test_losses_list:
            filehandle.write('%s\n' % listitem) 

    with open(save_path + 'local_avg_test_accuracy_list.txt', 'w') as filehandle:
        for listitem in local_avg_test_accuracy_list:
            filehandle.write('%s\n' % listitem) 

    # Plot Loss curve
    plt.figure()
    plt.title('Local Average Training Loss vs Communication rounds')
    plt.plot(range(len(local_avg_train_losses_list)), local_avg_train_losses_list, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_train_loss.png'.
                format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.message_passing_num, args.customize_test, args.cluster_num, args.frac))
    
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Local Average Test Loss vs Communication rounds')
    plt.plot(range(len(local_avg_test_losses_list)), local_avg_test_losses_list, color='k')
    plt.ylabel('Test Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_test_loss.png'.
                format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.message_passing_num, args.customize_test, args.cluster_num, args.frac))

    plt.figure()
    plt.title('Local Average Test Accuracy vs Communication rounds')
    plt.plot(range(len(local_avg_test_accuracy_list)), local_avg_test_accuracy_list, color='r')
    plt.ylabel('Test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_test_accuracy.png'.
                format(args.dataset, args.model, args.epochs,
                       args.iid, args.local_ep, args.message_passing_num, args.customize_test, args.cluster_num, args.frac))
