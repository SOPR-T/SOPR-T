import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter
import argparse
import time
import tqdm
import re
import json

import online_agent

from utils.constants import env_list
from utils.load_cluster import interact_with_data
from utils.metric import metric
import RankTransformerNet as RankTransformerNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--cluster_path", type=str, default=None)   # where is the training clusters
parser.add_argument("--validation_cluster", type=str, default=None) # where is the validation_clusters
parser.add_argument("--model_path", type=str, default=None) # where is the training policy models
parser.add_argument("--validation_path", type=str, default=None) # where is the validation policy models
parser.add_argument("--logdir", type=str, default=None) # where to save tensorboard
parser.add_argument("--save_dir", type=str, default=None)   # where to save rank model
parser.add_argument("--NUM_enc", type=int, default=64)  # average number of states in each cluster
parser.add_argument("--NUM_seg", type=int, default=256) # number of clusters
parser.add_argument("--NUM_enc_opt", type=int, default=64)  # embedding for each (s,a) pairs
# data type
parser.add_argument("--data_type", type=str, default='expert', choices=['medium', 'medium_expert', 'expert', 'random', 'full_replay', 'medium_replay'])
# mode:
# fixed -> fixed clusters
# random -> randomly sample states into clusters (k-means)
parser.add_argument("--training_mode", type=str, default='fixed', choices=['fixed', 'random'])
parser.add_argument("--training_states", type=str, default='16k', choices=['4k', '8k', '16k', '32k'])   # number of states for fixed mode
parser.add_argument("--score_mode", type=str, default='mean', choices=['mean', 'sort']) # scoring mode, how we get final score from clusters
parser.add_argument("--validation_mode", type=str, default='random', choices=['fixed', 'random'])   # validation mode
parser.add_argument("--validation_states", type=str, default='16k', choices=['4k', '8k', '16k', '32k'])   # number of states for fixed mode
parser.add_argument("--validation_repeat", type=int, default=10)    # how many batches for valiation in each epoch (fixed mode don't care about this)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--action_noise", type=float, default=0)    # noise when select actions

args = parser.parse_args()
# set random seed
setup_seed(args.seed)

state_dim = env_list[args.env]['state_dim']
action_dim = env_list[args.env]['action_dim']
max_action = float(env_list[args.env]['max_action'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setting = f"{args.env}_{args.data_type}"
print("----------------------")
print("env:", args.env)
print('seed:', args.seed)
print('data type:', args.data_type)
print("training mode:", args.training_mode)
print("validation mode:", args.validation_mode)
if args.validation_mode == 'random':
    print('validation repeat:', args.validation_repeat)
print('score mode:', args.score_mode)


NUM_features_enc = state_dim + action_dim  # enc input's dimension, state and action
NUM_encs = args.NUM_enc  # just a multiplier for calculating NUM_state
NUM_seg = args.NUM_seg
NUM_state = NUM_encs* NUM_seg  
NUM_enc_opt = args.NUM_enc_opt  # map [s,a] to a 64d vector
NUM_EPOCHS = args.epochs
if torch.cuda.is_available():
    print('use cuda')
    device = torch.device('cuda')
else:
    print('use cpu')
    device = torch.device('cpu')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.logdir:
    writer = SummaryWriter(args.logdir)
    print('log at ', args.logdir)
else:
    writer = SummaryWriter('log')
    print('log at log/')

policy_list = []  # training policies
# load policies
for Ind in range(30):
    policy = online_agent.OnlineAgent(state_dim=state_dim, action_dim=action_dim, algo='onlineSAC', device=device)
    policy.load(f"{args.model_path}/model_{Ind}.pt")

    policy_list.append(policy)

po_performance = np.load(f"{args.model_path}/performance.npy")
print('training policy number:', len(policy_list))
print('training policy:', po_performance)

val_policy_list = []  # training policies

# load policies
for Ind in range(10):
    policy = online_agent.OnlineAgent(state_dim=state_dim, action_dim=action_dim, algo='onlineSAC', device=device)
    policy.load(f"{args.validation_path}/model_{Ind}.pt")

    val_policy_list.append(policy)

val_po_performance = np.load(f"{args.validation_path}/performance.npy")
print('validation policy number:', len(val_policy_list))
print('validation policy:', val_po_performance)

# Load training clustered state
if args.training_mode == 'fixed':
    s_cluster = np.load(f"{args.cluster_path}/fixed/{args.training_states}/{args.env[:-3]}/{args.data_type}_list.npy", allow_pickle=True)  # elements this array are concatenated in the order of cluster1, cluster2, cluster3....
    s_clus_size = np.load(f"{args.cluster_path}/fixed/{args.training_states}/{args.env[:-3]}/{args.data_type}_size.npy", allow_pickle=True)  # in accordance with s_cluster
    sa4rank = interact_with_data(s_cluster, policy_list, torch.device('cpu'))
elif args.training_mode == 'random':
    state_clusters = []
    state_cluster_sizes = []
    for i in range(200):
        state_clusters.append(np.load(os.path.join(args.cluster_path, 'random', args.env[:-3].lower(), args.data_type, str(args.seed), f'cluster_{i}.npy'), allow_pickle=True))
        state_cluster_sizes.append(np.load(os.path.join(args.cluster_path, 'random', args.env[:-3].lower(), args.data_type, str(args.seed), f'cluster_size_{i}.npy'), allow_pickle=True))

# Load validation clustered state
if args.validation_mode == 'fixed':
    val_s_cluster = np.load(f"{args.validation_cluster}/fixed/{args.validation_states}/{args.env[:-3]}/{args.data_type}_list.npy", allow_pickle=True)  # elements this array are concatenated in the order of cluster1, cluster2, cluster3....
    val_s_clus_size = np.load(f"{args.validation_cluster}/fixed/{args.validation_states}/{args.env[:-3]}/{args.data_type}_size.npy", allow_pickle=True)  # in accordance with s_cluster
    val_sa4rank = interact_with_data(val_s_cluster, val_policy_list, torch.device('cpu'))
elif args.validation_mode == 'random':
    val_state_clusters = []
    val_state_cluster_sizes = []
    for i in range(200):
        val_state_clusters.append(np.load(os.path.join(args.validation_cluster, 'random', args.env[:-3].lower(), args.data_type, str(args.seed), f'cluster_{i}.npy'), allow_pickle=True))
        val_state_cluster_sizes.append(np.load(os.path.join(args.validation_cluster, 'random', args.env[:-3].lower(), args.data_type, str(args.seed), f'cluster_size_{i}.npy'), allow_pickle=True))

# Instantiate a RankTransformerNet
ranknet = RankTransformerNet.RankTransformer(NUM_enc_opt, NUM_features_enc, NUM_seg, False)
ranknet = ranknet.to(device)
torch.save(ranknet, f"{args.save_dir}/origin.pkl")

optimizer = torch.optim.Adam(ranknet.parameters(), lr = 0.001)
criterion = nn.BCELoss()
rank_threshold = 0.5  # 0.5 for prob outputs; 0 for score difference outputs

losses = []
losses_eval = []
rk_res_list = []

t0 = time.time()

loss_train_min = 1e8
coeff_train_max = -1
for epoch in range(NUM_EPOCHS):
    t1 = time.time()
    
    # prepare label
    num_po = len(policy_list)
    po_pair = int(num_po * (num_po - 1) / 2)
    p1_ind = np.zeros(po_pair, dtype=int)
    p2_ind = np.zeros(po_pair, dtype=int)
    label = torch.zeros(po_pair)
    ind = 0
    for i in range(num_po):
        for j in range(i+1, num_po):
            p1_ind[ind] = i
            p2_ind[ind] = j
            label[ind] = 1 if po_performance[i] > po_performance[j] else 0
            ind += 1
    ind = np.random.choice(po_pair, po_pair, replace=False) # shuffle
    p1_ind = p1_ind[ind]
    p2_ind = p2_ind[ind]
    label = label[ind]

    # prepare data
    if args.training_mode == 'fixed':
        tmp_sa = sa4rank
        tmp_clus_size = s_clus_size
    elif args.training_mode == 'random':
        tmp_sa = interact_with_data(state_clusters[epoch%len(state_clusters)], policy_list,device=torch.device('cpu'), noise_std=args.action_noise)
        tmp_clus_size = state_cluster_sizes[epoch%len(state_cluster_sizes)]

    # forward
    ranknet.train()
    scores = ranknet(tmp_sa.to(device), tmp_clus_size, mode=args.score_mode)

    score_1 = scores[p1_ind]
    score_2 = scores[p2_ind]
    prob = torch.sigmoid(score_1-score_2)

    loss = criterion(torch.squeeze(prob), label.to(device))  # BCE loss

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_train = loss.item()
    writer.add_scalar('Train/loss', loss_train, epoch)
    losses.append(loss_train)

    ranknet.eval()
    train_coeff, train_error_rate, train_error_pair, train_value_gap = metric(scores.cpu().detach().numpy().flatten(), po_performance)
    print(f"{epoch}/{NUM_EPOCHS} train coef:{train_coeff} train error:{train_error_rate} train loss:{loss_train}")
    writer.add_scalar("Train/corr", train_coeff, epoch)
    writer.add_scalar("Train/error_rate", train_error_rate, epoch)
    writer.add_scalar("Train/value_gap", train_value_gap, epoch)

    # validation
    ranknet.eval()
    # prepare label
    num_po = len(val_policy_list)
    po_pair = int(num_po * (num_po - 1) / 2)
    p1_ind = np.zeros(po_pair, dtype=int)
    p2_ind = np.zeros(po_pair, dtype=int)
    label = torch.zeros(po_pair)
    ind = 0
    for i in range(num_po):
        for j in range(i+1, num_po):
            p1_ind[ind] = i
            p2_ind[ind] = j
            label[ind] = 1 if val_po_performance[i] > val_po_performance[j] else 0
            ind += 1
    ind = np.random.choice(po_pair, po_pair, replace=False)
    p1_ind = p1_ind[ind]
    p2_ind = p2_ind[ind]
    label = label[ind]

    if args.validation_mode == 'random':
        score_list = []
        loss_list = []
        ind = np.random.choice(len(val_state_clusters), args.validation_repeat)
        for i in range(args.validation_repeat):
            tmp_sa = interact_with_data(val_state_clusters[ind[i]], val_policy_list, device=torch.device('cpu'))
            tmp_clus_size = val_state_cluster_sizes[ind[i]]
            scores = ranknet(tmp_sa.to(device), tmp_clus_size, mode=args.score_mode).cpu().detach().flatten()
            score_list.append(scores)
            score_1 = scores[p1_ind]
            score_2 = scores[p2_ind]
            prob = torch.sigmoid(score_1-score_2)

            loss = criterion(prob, label)  # BCE loss
            loss_list.append(loss.item())
        scores = torch.stack(score_list).transpose(0, 1)
        scores = scores.mean(1).cpu().numpy().flatten()
        validation_loss = sum(loss_list)/len(loss_list)
    elif args.validation_mode == 'fixed':
        tmp_sa = val_sa4rank
        tmp_clus_size = val_s_clus_size
        scores = ranknet(tmp_sa.to(device), tmp_clus_size, mode=args.score_mode).cpu().detach().flatten()
        score_1 = scores[p1_ind]
        score_2 = scores[p2_ind]
        prob = torch.sigmoid(score_1-score_2)

        loss = criterion(prob, label)  # BCE loss
        validation_loss = loss.item()
    coeff, error_rate, error_pair, value_gap = metric(scores, val_po_performance)
    # print(f"{epoch}/{NUM_EPOCHS} validation coef:{coeff} validation error:{error_rate} ")
    writer.add_scalar("Validation/corr", coeff, epoch)
    writer.add_scalar("Validation/error_rate", error_rate, epoch)
    writer.add_scalar("Validation/value_gap", value_gap, epoch)
    writer.add_scalar("Validation/loss", validation_loss, epoch)

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}, Training_error_rate: {:.10f}, timecost: {:.2f}'.format(epoch + 1, NUM_EPOCHS, loss_train, error_rate, time.time() - t1))

    if coeff >= coeff_train_max:
        print('--------------------------')
        print(f'new best corr epoch:{epoch}')
        print('--------------------------')
        coeff_train_max = coeff
        torch.save(ranknet, os.path.join(args.save_dir, 'corr_best.pkl'))
    if validation_loss <= loss_train_min:
        print('--------------------------')
        print(f'new best loss epoch:{epoch}')
        print('--------------------------')
        loss_train_min = validation_loss
        torch.save(ranknet, os.path.join(args.save_dir, 'loss_best.pkl'))

print('total time:', time.time()-t0)

