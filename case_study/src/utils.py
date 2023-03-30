import copy
import torch
# from sklearn.cluster import KMeans
import numpy as np
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid,noniid_server_graph, mnist_iid, mnist_noniid, svhn_iid, svhn_noniid, iid, noniid
from options import args_parser

args = args_parser()
device = args.device
cos = torch.nn.CosineSimilarity(dim=0)


def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num



def similarity_score(dict_new, dict_base, example_model): 
    # dict_new is one graph, index is node, value is model param dict
    # dict_base is one graph as well, index is node, value is model param dict
    init_model_new = copy.deepcopy(example_model.to(device))
    init_model_base = copy.deepcopy(example_model.to(device))

    similarity_score_node = []
    for i in range(len(dict_new)):
        init_model_new.load_state_dict(dict_new[i])
        para_new_list = []
        for param_new in init_model_new.parameters():
            para_new_list.extend(torch.flatten(param_new).cpu().detach().numpy())
        simi_temp = []
        for j in range(len(dict_base)):
            init_model_base.load_state_dict(dict_base[j])
            para_base_list = []
            for param_base in init_model_base.parameters():
                para_base_list.extend(torch.flatten(param_base).cpu().detach().numpy())
            simi = cos(torch.Tensor(para_new_list).float(), torch.Tensor(para_base_list).float())
            simi_temp.append(simi)
        similarity_score_node.append(simi_temp)
    return similarity_score_node
            


    



    # for value_new in dict_new.values():
    #     para_new_list = []
    #     for param in value_new.parameters():
    #         para_new_list.extend(torch.flatten(param).cpu().detach().numpy())
    # for value_base in dict_base.values():
    #     para_base_list = []
    #     for param in value_base.parameters():
    #         para_base_list.extend(torch.flatten(param).cpu().detach().numpy())
    # for i in range(len(dict_new)):
    #     temp_list = []
    #     for j in range(len(dict_base)):
    #         temp_list.append(cos(torch.Tensor(para_new_list[i]).float(),torch.Tensor(para_base_list[j]).float()).item())
    #     # temp_normalize_list = [(k - min(temp_list))/(max(temp_list)-min(temp_list)) for k in temp_list]
    #     similarity_cluster_score.append(temp_list)
    # return similarity_cluster_score # a list [[][][][]] the first list is the index = 0 of new dict


def j_score(dict_new, dict_base): # base is the last round, dict_new is this round
    j_cluster_score = []
    for value_new in dict_new.values():
        temp_list = []
        for value_base in dict_base.values():
            intersection = len(list(set(value_new).intersection(value_base)))
            union = (len(value_new) + len(value_base)) - intersection
            j_ = float(intersection) / union
            temp_list.append(j_)
        # temp_normalize_list = [(i - min(temp_list))/(max(temp_list)-min(temp_list)) for i in temp_list]
        j_cluster_score.append(temp_list)
    return j_cluster_score # a list [[][][][]] the first sub list is the index = 0 of new dict


def adj_matrix_converter_flatten(model_list):
    num = len(model_list)
    model_copy_list = [copy.deepcopy(item) for item in model_list]
    local_model_flatten_list = [] # length = number of cluster
    for each_model in model_copy_list:
        temp_list = []
        for param in each_model.parameters():
            temp_list.extend(torch.flatten(param).cpu().detach().numpy())
        local_model_flatten_list.append(temp_list)
    adjacency_matrix = np.ones((num, num))

    
    for i in range(num):
        for j in range(num):
            adjacency_matrix[i][j] = cos(torch.Tensor(local_model_flatten_list[i]).float(),torch.Tensor(local_model_flatten_list[j]).float()).item()

    # adjacency_matrix_map = np.ones((num, num))
    # for i in range(num):
    #     for j in range(num):
    #         if adjacency_matrix[i][j] >= (sum(adjacency_matrix[i])-1)/(num-1) or i==j:
    #             adjacency_matrix_map[i][j] = 1
    #         elif adjacency_matrix[i][j] < (sum(adjacency_matrix[i])-1)/(num-1) and i != j:
    #             adjacency_matrix_map[i][j] = 0
    adjacency_matrix_map = np.zeros((num, num))
    for j in range(num):
        adjacency_matrix_map[:,j] = adjacency_matrix[:,j]/sum(adjacency_matrix[:,j])
    return adjacency_matrix, adjacency_matrix_map





def average_weights(w):

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

########################### print exp parameters ##################################
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Communication Rounds   : {args.epochs}')
    print(f'    K-means cluster number   : {args.cluster_num}')
    print(f'    Message passing number   : {args.message_passing_num}\n')
    print(f'    Check freq   : {args.check_freq}\n')
    if args.special_user:
        print('    have special users')
    else:
        print('    have no special users')
    if args.customize_test:
        print('    customize test')
    else:
        print('    general test')

    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    if args.special_user:
        print(f'    Number of users    : {args.cluster_num * args.number_client_node}')
    else:
        print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    # print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local training Epochs : {args.local_ep}\n')
    return


# TODO to create a new function return normal and special users


# def get_datasets_with_special(args):

#     if args.dataset == 'cifar':
#         data_dir = '../data/cifar'
#         apply_transform = transforms.Compose(
#             [transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#             ]
#         )

#         train_dataset = datasets.CIFAR10(data_dir, train = True, download = True, transform = apply_transform)
#         test_dataset = datasets.CIFAR10(data_dir, train = False, download = True, transform = apply_transform)
#         if args.iid:
#             dict_users_train,  dict_users_test = cifar_iid(train_dataset, args.num_users)
#         else:
#             dict_users_train, dict_users_test = cifar_noniid(train_dataset, args.num_users)


#     if args.dataset == 'svhn':
#         data_dir = '../data/svhn'
#         apply_transform = transforms.Compose(
#             [transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#             ]
#         )
#         train_dataset = datasets.SVHN(data_dir, split = 'train', download = True, transform = apply_transform)
#         test_dataset = datasets.SVHN(data_dir, split = 'test', download = True, transform = apply_transform)
#         if args.iid:
#             dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
#             # dict_users_train,  dict_users_test = svhn_iid(train_dataset, args.num_users)
#         else:
#             dict_users_train, dict_users_test = noniid(train_dataset, args.num_users, args)

#     elif args.dataset == 'mnist':

#         data_dir = '../data/mnist'


#         apply_transform = transforms.Compose(
#             [transforms.ToTensor(),
#                 transforms.Normalize((0.1307,),(0.3081,))
#             ]
#         )

#         train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)
#         test_dataset = datasets.MNIST(data_dir, train = False, download = True, transform = apply_transform)

#         if args.iid:
#             dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
#         else:
#             dict_users_train, dict_users_test = noniid(train_dataset, args.num_users, args)    
    
 
#     return train_dataset, test_dataset, dict_users_train, dict_users_test


def get_datasets(args):

    if args.dataset == 'cifar':
        data_dir = '../data/cifar'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )

        train_dataset = datasets.CIFAR10(data_dir, train = True, download = True, transform = apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train = False, download = True, transform = apply_transform)
        if args.iid:
            dict_users_train,  dict_users_test = cifar_iid(train_dataset, args.num_users)
        else:
            dict_users_train, dict_users_test = cifar_noniid(train_dataset, args.num_users)


    if args.dataset == 'svhn':
        data_dir = '../data/svhn'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )
        train_dataset = datasets.SVHN(data_dir, split = 'train', download = True, transform = apply_transform)
        test_dataset = datasets.SVHN(data_dir, split = 'test', download = True, transform = apply_transform)
        if args.iid:
            dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
            # dict_users_train,  dict_users_test = svhn_iid(train_dataset, args.num_users)
        else:
            if args.special_user:
                dict_users_train, dict_users_test,temp_index_dict, dict_user_id_cluster_id, special_users_label_dict = noniid_server_graph(train_dataset, args.number_client_node * args.cluster_num, args)
            else:
                dict_users_train, dict_users_test = noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':

        data_dir = '../data/mnist'


        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ]
        )

        train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)
        test_dataset = datasets.MNIST(data_dir, train = False, download = True, transform = apply_transform)

        if args.iid:
            dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
        else:
            if args.special_user:
                dict_users_train, dict_users_test,temp_index_dict, dict_user_id_cluster_id, special_users_label_dict = noniid_server_graph(train_dataset, args.number_client_node * args.cluster_num, args)
            else:
                dict_users_train, dict_users_test = noniid(train_dataset, args.num_users)    
        # dict_user_id_cluster_id
    
    # elif args.dataset == 'mnist' or 'fmnist':
    #     if args.dataset == 'mnist':
    #         data_dir = '../data/mnist'
    #     else:
    #         data_dir = '../data/fmnist'

    #     apply_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #             transforms.Normalize((0.1307,),(0.3081,))
    #         ]
    #     )

    #     train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)
    #     test_dataset = datasets.MNIST(data_dir, train = False, download = True, transform = apply_transform)

    #     if args.iid:
    #         dict_users_train,  dict_users_test = iid(train_dataset, args.num_users)
    #     else:
    #         dict_users_train, dict_users_test = noniid(train_dataset, args.num_users)
    if args.special_user:
        return train_dataset, test_dataset, dict_users_train, dict_users_test, temp_index_dict, dict_user_id_cluster_id, special_users_label_dict
    else:
        return train_dataset, test_dataset, dict_users_train, dict_users_test