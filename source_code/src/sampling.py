import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
np.random.seed(0)

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

# def noniid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users_train, dict_users_test, all_idxs = {}, {},[i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users_train[i] = set(np.random.choice(all_idxs, int(num_items *0.8),
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users_train[i])
#         dict_users_test[i] = set(np.random.choice(all_idxs, int(num_items *0.2),
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users_train[i])                      
#     return dict_users_train, dict_users_test

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
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
"""
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


def svhn_noniid(dataset, num_users, args):
    """
    Sample non-I.I.D client data from svhn dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = args.num_shards, int(len(dataset)/num_users/2)
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
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test


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

def cifar_non_iid_test(dataset, num_users, args):
    # num_shards, num_imgs = 200, 250
    num_shards, num_imgs = int(args.num_shards), int(200 * 250 / args.num_shards)

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

    idxs_label = idxs_labels[1, :]
    dict_users_train_label = {i: np.array([]) for i in range(num_users)}
    dict_users_test_label = {i: np.array([]) for i in range(num_users)}

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            dict_users_train_label[i] = np.concatenate(
                (dict_users_train[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][0:int(num_imgs * 0.8)]), axis=0)
            
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)

            dict_users_test_label[i] = np.concatenate(
                (dict_users_test[i], idxs_label[rand*num_imgs:(rand+1)*num_imgs][(int(num_imgs * 0.8) - num_imgs):]), axis=0)
    # num_label = len(list(set(idxs_label)))
    # label_dist = [[] for _ in range(num_label)]
    # label_dist is a list, length = 10, in each list, is the client id

    # A = np.zeros((num_users, num_users))
    # num_label = len(set(labels))
    # label_dist = [[] for _ in range(num_label)]

    # for i in range(num_label):
    #     for key,value in dict_users_train_label.items():
    #         for label in value:
    #             if label == i:
    #                 label_dist[i].append(key)
    # link_list = []
    # for user_arr in label_dist:
    #     for user_a in user_arr:
    #         for user_b in user_arr:
    #             link_list.append([user_a, user_b])
    # link_sample = list(range(len(link_list)))
    # link_idx = np.random.choice(link_sample, int(args.edge_frac * len(link_list)), replace=False)
    # for idx in link_idx:
    #     # A[link_list[idx][0], link_list[idx][1]] = A[link_list[idx][0], link_list[idx][1]] + 1
    #     A[link_list[idx][0], link_list[idx][1]] = 1

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
    # idxs_labels is a two dimensional array
    # the first row is the row index
    # the second row is the label
    # like:
    # [[29513, 16836, 32316, ..., 36910, 21518, 25648],
    # [    0,     0,     0, ...,     9,     9,     9]]

    idxs = idxs_labels[0, :]
    # idxs is the second row of idxs_labels


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # rand_set is 200 shard里面选两个
        idx_shard = list(set(idx_shard) - rand_set)
        # idx_shard 是除了rand_set之外的剩下的，为下一轮做准备
        for rand in rand_set:
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand*num_imgs:(rand+1)*num_imgs][0:240]), axis=0)

            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs[rand*num_imgs:(rand+1)*num_imgs][-60:]), axis=0)           
    return dict_users_train, dict_users_test




# def mnist_noniid_unequal(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30

#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)

#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:

#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         random_shard_size = random_shard_size-1

#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#     else:

#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#     return dict_users






if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)