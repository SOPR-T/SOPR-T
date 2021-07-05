# get actions for states of each cluster and concate
import torch
import numpy as np


def interact_with_data(s_cluster, pol_list, device=torch.device('cuda'), noise_std=0):
    if type(s_cluster) is list:
        NUM_seg = len(s_cluster)
    else:
        NUM_seg = s_cluster.shape[0]
    num_pol = len(pol_list)
    NUM_state = 0
    # loop among different clusters
    # we get (num_state, (state_dim+action_dim)*num_policy)
    for seg_ind in range(int(NUM_seg)):
        # loop among different buffers
        s_tmp_num = s_cluster[seg_ind]  # shape = [clus_size, 17]
        NUM_state += s_tmp_num.shape[0]
        s_tmp = torch.Tensor(s_tmp_num).to(device)

        # for each cluster, stack state and action, we get (cluster_size, (state_dim+action_dim)*num_policy)
        for po_index in range(num_pol):
            po = pol_list[po_index]
            a_tmp = po.select_action_buffer(s_tmp)
            if not type(a_tmp) is np.ndarray:
                a_tmp = a_tmp.cpu().data.numpy()  # size: (cluster_size, action_dim)
            if noise_std > 0:
                a_tmp += noise_std * np.random.randn(a_tmp.shape[0], a_tmp.shape[1])


            if po_index == 0:
                sa4rank = np.hstack((s_tmp_num, a_tmp))  # stack state and action, expand dim=1 to state_dim+action_dim
            else:
                sa4rank = np.hstack((sa4rank, np.hstack((s_tmp_num, a_tmp))))
        if seg_ind == 0:
            sa4rank_final = sa4rank
        else:
            sa4rank_final = np.vstack((sa4rank_final, sa4rank))
        

    sa4rank = torch.Tensor(sa4rank_final) # (num_state, (state_dim+action_dim)*num_policy)
    # reshape we get (num_policy*num_state, state_dim+action_dim)
    '''
        state_1, policy_1 | state action
        state_1, policy_2 | state action
        .
        .
        .
        state_2, policy_1 | state action
        .
        .
        .
        state_N, policy_M | state action 
    '''
    sa4rank = torch.reshape(sa4rank,
                                    (int(num_pol * NUM_state), -1))

    return sa4rank.to(device)