import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return torch.tensor(image), torch.tensor(label)


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test


def noniid(dataset, num_users, args):
    """
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test



# def noniid(dataset, num_users, args):
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users_train = {i: np.array([]) for i in range(num_users)}
#     dict_users_test = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()


#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#     special_users = [1,2,3,4,5,6,7,8,9,10]
#     temp_label_dict = {q:[] for q in range(10)}
#     temp_index = list(idxs_labels[0,:])
#     temp_label = list(idxs_labels[1,:])
#     for j in range(len(temp_label)):
#         for key in temp_label_dict.keys():
#             if temp_label[j] == key:
#                 temp_label_dict[key].append(j)
#     temp_index_dict = {q:[] for q in range(10)}
#     for key in temp_index_dict.keys():
#         temp_index_dict[key] = [temp_index[item] for item in temp_label_dict[key]]
#     # in temp_index_dict, key is the label, value is a list of index


#     special_users_label_dict = {
#         1: [0,1],
#         2: [4],
#         3: [7],
#         4: [0,2],
#         5: [3,5],
#         6: [7,5],
#         7: [2,3],
#         8: [5,6],
#         9:[6],
#         10:[3]
#     }
#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         if i not in special_users:
#             rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
                
#                 dict_users_train[i] = np.concatenate(
#                     (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
#                 dict_users_test[i] = np.concatenate(
#                     (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)
#         else:
#             user_label_list = special_users_label_dict[i]
#             for user_label in user_label_list:
#                 dict_users_train[i] = np.concatenate((dict_users_train[i], np.array(temp_index_dict[user_label][0:240])), axis = 0)
#                 dict_users_test[i] = np.concatenate((dict_users_test[i], np.array(temp_index_dict[user_label][-60:])), axis = 0)
        
#     return dict_users_train, dict_users_test



def noniid_server_graph(dataset, num_users, args):
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_users = args.number_client_node * args.cluster_num
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_user_id_label_id ={i: set() for i in range(num_users)}
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    if args.dataset == 'svhn':
        labels = dataset.labels[0:len(idxs)]
    else:
        labels = dataset.train_labels.numpy()
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    temp_label_dict = {q:[] for q in range(10)}
    temp_index = list(idxs_labels[0,:])
    temp_label = list(idxs_labels[1,:])
    for j in range(len(temp_label)):
        for key in temp_label_dict.keys():
            if temp_label[j] == key:
                temp_label_dict[key].append(j)
    temp_index_dict = {q:[] for q in range(10)}
    for key in temp_index_dict.keys():
        temp_index_dict[key] = [temp_index[item] for item in temp_label_dict[key]]

    # in temp_index_dict, key is the label, value is a list of index
    if args.cluster_num == 3:
        special_users_label_dict = {
            1: [0,1,2], # first 20 clients
            2: [2,3,4],
            3: [4,5,6]
        }
    elif  args.cluster_num == 4:
        special_users_label_dict = {
            1: [0,1,2], 
            2: [2,3,4],
            3: [4,5,6],
            4: [3,7,8]
        }
    elif args.cluster_num == 5:
        special_users_label_dict = {
            1: [0,1,2], 
            2: [2,3,4],
            3: [4,5,6],
            4: [3,7,8],
            5: [9,8]
        }        

    elif args.cluster_num == 6:
        special_users_label_dict = {
            1: [0,1,9], # first 20 clients
            2: [1,8],
            3: [0,2],
            4: [5,9],
            5: [2,3],
            6: [5,6]
        } # index is the graph node, value is the label list
        # # number_node
    
    elif args.cluster_num == 7:
        special_users_label_dict = {
            1: [0,1,9],
            2: [1,8],
            3: [0,2],
            4: [4,5,9],
            5: [4,6,9],
            6: [2,3],
            7: [5,6]
        } # index is the user, value is the label list


    
    # for dict_user_id_cluster_id, key is the client id, value is is the cluster id
    dict_user_id_cluster_id = {}
    for key in special_users_label_dict.keys(): # key is the graph node, for each node we have 20 clients
        for j in range(args.number_client_node): # j is 0-19
            temp_index_id = (key-1) * args.number_client_node + j # 0-1-2-3-4-..119
            # print(temp_index_id)
            dict_user_id_cluster_id[temp_index_id] = key
            for q in range(len(special_users_label_dict[key])): # q is index of that value list
                temp_length = int(len(temp_index_dict[special_users_label_dict[key][q]])/args.number_client_node)
                # for each label, each client should have how many data
                current_value_train = np.array(temp_index_dict[special_users_label_dict[key][q]][j*(temp_length):int(j*(temp_length) + (temp_length)*0.8)])
                current_value_test = np.array(temp_index_dict[special_users_label_dict[key][q]][int(j*(temp_length) + (temp_length)*0.8):(j+1)*(temp_length)])

                dict_users_train[temp_index_id] = np.concatenate((dict_users_train[temp_index_id], current_value_train),axis = 0)
                dict_users_test[temp_index_id] = np.concatenate((dict_users_train[temp_index_id], current_value_test),axis = 0)
    
    # index_label_dict = dict(zip(idxs, list(labels)))
    # key is index, value is label
    index_label_dict_made = {}
    for key, value in temp_index_dict.items():
        for item in value:
            index_label_dict_made[item] = key
    
    
    for key,value in dict_users_train.items():
        for item in value:
            dict_user_id_label_id[key].add(index_label_dict_made[item])
    # dict_users_train key is user_id, value is row index
    # temp_index_dict key is the label, value is a list of index
    # we want key is user_id, value is label
    #

    # for dict_user_id_cluster_id, key is the client id, value is is the cluster id
    

    return dict_users_train, dict_users_test, dict_user_id_label_id, dict_user_id_cluster_id, special_users_label_dict



def noniid_special(dataset, num_users, args):
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    if args.dataset == 'svhn':
        labels = dataset.labels[0:len(idxs)]
    else:
        labels = dataset.train_labels.numpy()
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    special_users = [1,2,3,4,5,6,7,8,9,10]
    temp_label_dict = {q:[] for q in range(10)}
    temp_index = list(idxs_labels[0,:])
    temp_label = list(idxs_labels[1,:])
    for j in range(len(temp_label)):
        for key in temp_label_dict.keys():
            if temp_label[j] == key:
                temp_label_dict[key].append(j)
    temp_index_dict = {q:[] for q in range(10)}
    for key in temp_index_dict.keys():
        temp_index_dict[key] = [temp_index[item] for item in temp_label_dict[key]]
    # in temp_index_dict, key is the label, value is a list of index
    special_users_label_dict = {
        1: [0,1,9],
        2: [1,8],
        3: [0,2],
        4: [5,9],
        5: [2,3],
        6: [5,6]
    } # index is the user, value is the label list
    # special_users_label_dict = {
    #     1: [0,1,9],
    #     2: [1,8],
    #     3: [0,2],
    #     4: [4,5,9],
    #     5: [4,6,9],
    #     6: [2,3],
    #     7: [5,6]
    # } # index is the user, value is the label list

    for i in range(num_users):
        if i not in special_users:
            rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
                dict_users_test[i] = np.concatenate(
                    (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)
        else:
            user_label_list = special_users_label_dict[i]
            for user_label in user_label_list:
                dict_users_train[i] = np.concatenate((dict_users_train[i], np.array(temp_index_dict[user_label][0:240])), axis = 0)
                dict_users_test[i] = np.concatenate((dict_users_test[i], np.array(temp_index_dict[user_label][-60:])), axis = 0)
        
    return dict_users_train, dict_users_test






















# def noniid_with_special(dataset, num_users, args):
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users_train = {i: np.array([]) for i in range(num_users)}
#     dict_users_test = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     if args.dataset == 'svhn':
#         labels = dataset.labels[0:len(idxs)]
#     else:
#         labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
            
#             dict_users_train[i] = np.concatenate(
#                 (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
#             dict_users_test[i] = np.concatenate(
#                 (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
#     return dict_users_train, dict_users_test



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test



def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test

# should return two_dictionaries, one is for train, one is for test
def mnist_noniid(dataset, num_users):

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test

def svhn_noniid(dataset, num_users):

    num_shards, num_imgs = 2 * num_users, int(len(dataset)/num_users/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.labels[0:len(idxs)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test



def svhn_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])
        dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users_train[i])                      
    return dict_users_train, dict_users_test


def svhn_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from svhn dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 2 * num_users, int(len(dataset)/num_users/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.labels[0:len(idxs)]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # how many classes - 2 
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)