import numpy as np
import torch
import argparse
import time
from utils.constants import env_list
from utils.load_cluster import interact_with_data
import os
import offline_agent
import online_agent
from utils.metric import metric


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--seed1", type=int, default=0) # tell the seed used to train ranknet
parser.add_argument("--ranking_model", type=str, default=None)
parser.add_argument("--save_dir", type=str, default='results')
parser.add_argument("--ranknet_type", type=str, default='corr') # use model saved according to what metric: corr/loss
parser.add_argument("--cluster_path", type=str, default=None)   # where is the cluster
parser.add_argument("--repeat", type=int, default=200)
parser.add_argument("--model_path1", type=str, default=None) # where is the policy model
parser.add_argument("--model_path2", type=str, default=None)
parser.add_argument("--data_type", type=str, choices=['medium', 'expert', 'random', 'full_replay', 'medium_expert', 'medium_replay'])
# mode:
# fixed -> fixed clusters
# random -> randomly sample states into clusters (k-means)
# sequential -> divide all states into sequential clusters
# full -> divide all states into 256 clusters and randomly sample states in each cluster
parser.add_argument("--eval_mode", type=str, default='fixed', choices=['fixed', 'random', 'sequential', 'full'])
parser.add_argument("--test_states", type=str, default='16k', choices=['4k', '8k', '16k', '32k'])   # number of states for fixed mode
parser.add_argument("--score_mode", type=str, default='mean')
parser.add_argument("--final_score", type=str, default='mean', choices=['mean', 'medium'])  # how to get final score from different batches
args = parser.parse_args()

# set random seed
setup_seed(args.seed)

state_dim = env_list[args.env]['state_dim']
action_dim = env_list[args.env]['action_dim']
max_action = float(env_list[args.env]['max_action'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setting = f"{args.env}_{args.data_type}"
print("----------------------")
print("env: ", args.env)
print('data type:', args.data_type)
print('eval mode:', args.eval_mode)
print('ranknet type:', args.ranknet_type)
print('model path 1:', args.model_path1)
print('model path 2:', args.model_path2)


rankingModel_path = f"{args.ranking_model}/{args.ranknet_type}_best.pkl"

# # Load policies
t = time.time()
policy_list1 = []
algos = np.load(os.path.join(args.model_path1, 'test', args.env[:-3].lower(), 'set1', args.data_type, 'algo.npy'))
po_performance1 = np.load(os.path.join(args.model_path1, 'test', args.env[:-3].lower(), 'set1', args.data_type, 'performance.npy'))
for i in range(10):
    tmp_policy = offline_agent.OfflineAgent(state_dim=state_dim, action_dim=action_dim, algo=algos[i], device=device)
    tmp_policy.load(os.path.join(args.model_path1, 'test', args.env[:-3].lower(), 'set1', args.data_type, f'model_{i}.pt'))
    policy_list1.append(tmp_policy)

print('test set I:', po_performance1)

# Load policies
policy_list2 = []  # training policies
po_performance2 = np.load(os.path.join(args.model_path2, 'test', args.env[:-3].lower(), 'set2', 'performance.npy'))
for Ind in range(10):
    policy = online_agent.OnlineAgent(state_dim=state_dim, action_dim=action_dim, algo='onlineSAC', device=device)
    policy.load(f"{args.model_path2}/test/{args.env[:-3].lower()}/set2/model_{Ind}.pt")  # uniform checkpoints  [int(Ind*100/NUM_policies)]

    policy_list2.append(policy)

print('test set II:', po_performance2)

print('policy load finished, time spent:{:.2f}'.format(time.time()-t))

# Load clustered state
t = time.time()
# Load clustered state
if args.eval_mode == 'fixed':
    s_cluster = np.load(f"{args.cluster_path}/fixed/{args.test_states}/{args.env[:-3]}/{args.data_type}_list.npy", allow_pickle=True)  # elements this array are concatenated in the order of cluster1, cluster2, cluster3....
    s_clus_size = np.load(f"{args.cluster_path}/fixed/{args.test_states}/{args.env[:-3]}/{args.data_type}_size.npy", allow_pickle=True)  # in accordance with s_cluster
    sa4rank1 = interact_with_data(s_cluster, policy_list1, torch.device('cpu'))
    sa4rank2 = interact_with_data(s_cluster, policy_list2, torch.device('cpu'))
elif args.eval_mode == 'random':
    state_clusters = []
    state_cluster_sizes = []
    cluster_ind = np.arange(200)
    np.random.shuffle(cluster_ind)
    cluster_ind = cluster_ind.tolist()
    for i in cluster_ind[:args.repeat]:
        state_clusters.append(np.load(os.path.join(args.cluster_path, 'random', args.env[:-3].lower(), args.data_type, str(args.seed1), f'cluster_{i}.npy'), allow_pickle=True))
        state_cluster_sizes.append(np.load(os.path.join(args.cluster_path, 'random', args.env[:-3].lower(), args.data_type, str(args.seed1), f'cluster_size_{i}.npy'), allow_pickle=True))
    print('clusters:', len(state_clusters))
print('cluster load finished, time spent:{:.2f}s'.format(time.time()-t))

t = time.time()
ranknet = torch.load(rankingModel_path)
ranknet.eval()
print('ranknet load finished, time spent:{:.2f}'.format(time.time()-t))

rank_threshold = 0

# print(f"-----------------------Eval policy type: {'MIXED'} ---------------")
# print(f"-----------------------Eval policy mode: {args.eval_mode} ---------------")
num_eval_policies = len(policy_list1)
t = time.time()
# prepare data
if args.eval_mode == 'fixed':
    tmp_sa1 = sa4rank1
    tmp_sa2 = sa4rank2
    tmp_clus_size = s_clus_size
    scores1 = ranknet(tmp_sa1.to(device), tmp_clus_size, mode=args.score_mode).cpu().detach().flatten()
    scores2 = ranknet(tmp_sa2.to(device), tmp_clus_size, mode=args.score_mode).cpu().detach().flatten()
elif args.eval_mode == 'random':
    score_list1 = []
    score_list2 = []
    for i in range(len(state_clusters)):
        tmp_sa1 = interact_with_data(state_clusters[i], policy_list1,device=torch.device('cpu'))
        tmp_sa2 = interact_with_data(state_clusters[i], policy_list2,device=torch.device('cpu'))
        tmp_clus_size = state_cluster_sizes[i]
        scores1 = ranknet(tmp_sa1.to(device), tmp_clus_size, mode=args.score_mode).cpu().detach().flatten()
        score_list1.append(scores1)
        scores2 = ranknet(tmp_sa2.to(device), tmp_clus_size, mode=args.score_mode).cpu().detach().flatten()
        score_list2.append(scores2)
    scores1 = torch.stack(score_list1).transpose(0, 1)
    scores2 = torch.stack(score_list2).transpose(0, 1)
    if args.final_score == 'mean':
        scores1 = scores1.mean(1)
        scores2 = scores2.mean(1)
    elif args.final_score == 'medium':
        scores1 = scores1[:, scores1.shape[1]//2]
        scores2 = scores2[:, scores2.shape[1]//2]

print('scores get, time spent:{:.2f}'.format(time.time()-t))

coeff, error_rate, error_pair, value_gap = metric(scores1, po_performance1)
print('test set I:')
v_max_gt = po_performance1.max()
v_norm = v_max_gt - po_performance1.min()
regret3 = (v_max_gt - po_performance1[np.max(np.argsort(scores1.cpu().numpy())[-3:])]) / v_norm
regret3 = round(regret3, 4)
print('coeff:', round(coeff,4), ', error rate:', round(error_rate, 4), ', value gap:', round(value_gap / po_performance1.max(), 4), ', regret@3:', regret3, ', rank:', np.argsort(scores1).tolist())
with open(os.path.join(args.save_dir, f'{args.env[:-3]}_{args.data_type}_{args.ranknet_type}_{args.repeat}_{args.seed1}_{args.seed}_I.txt'), 'w') as f:
    f.write('coeff: {:.4f}, error rate: {:.4f}, value gap: {:.4f}, regret@3: {:.4f}, rank: {}'.format(coeff, error_rate, value_gap / po_performance1.max(), regret3, np.argsort(scores1).tolist()))
np.save(os.path.join(args.save_dir, f'{args.env[:-3]}_{args.data_type}_{args.ranknet_type}_{args.repeat}_{args.seed1}_{args.seed}_I_score.npy'), scores1.cpu().numpy())

coeff, error_rate, error_pair, value_gap = metric(scores2, po_performance2)
print('test set II:')
v_max_gt = po_performance2.max()
v_norm = v_max_gt - po_performance2.min()
regret3 = (v_max_gt - po_performance2[np.max(np.argsort(scores2.cpu().numpy())[-3:])]) / v_norm
regret3 = round(regret3, 4)
print('coeff:', round(coeff,4), ', error rate:', round(error_rate, 4), ', value gap:', round(value_gap / po_performance2.max(), 4), ', regret@3:', regret3, ', rank:', np.argsort(scores2).tolist())
with open(os.path.join(args.save_dir, f'{args.env[:-3]}_{args.data_type}_{args.ranknet_type}_{args.repeat}_{args.seed1}_{args.seed}_II.txt'), 'w') as f:
    f.write('coeff: {:.4f}, error rate: {:.4f}, value gap: {:.4f}, regret@3: {:.4f}, rank: {}'.format(coeff, error_rate, value_gap / po_performance2.max(), regret3, np.argsort(scores2).tolist()))
np.save(os.path.join(args.save_dir, f'{args.env[:-3]}_{args.data_type}_{args.ranknet_type}_{args.repeat}_{args.seed1}_{args.seed}_II_score.npy'), scores2.cpu().numpy())





