import torch
import torch.nn.functional as F
import pdb
import numpy as np
    
def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

def cvae_loss(pred_goal, pred_traj, target, best_of_many=True):
        '''
        CVAE loss use best-of-many
        Params:
            pred_goal: (Batch, K, pred_dim)
            pred_traj: (Batch, T, K, pred_dim)
            target: (Batch, T, pred_dim)
            best_of_many: whether use best of many loss or not
        Returns:

        '''
        K = pred_goal.shape[1]
        target = target.unsqueeze(2).repeat(1, 1, K, 1)
    
        # select bom based on  goal_rmse
        goal_rmse = torch.sqrt(torch.sum((pred_goal - target[:, -1, :, :])**2, dim=-1))
        traj_rmse = torch.sqrt(torch.sum((pred_traj - target)**2, dim=-1)).sum(dim=1)
        if best_of_many:
            best_idx = torch.argmin(goal_rmse, dim=1)
            loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        else:
            loss_goal = goal_rmse.mean()
            loss_traj = traj_rmse.mean()
        
        return loss_goal, loss_traj

def human_constraint_loss( pred_traj, target, cfg):
    """
    Params:
            pred_goal: (Batch, K, pred_dim)
            pred_traj: (Batch, T, K, pred_dim)
            target: (Batch, T, pred_dim)
    """

    kpoint = {1: [0,1], 2:[2,3], 3:[4,5], 4:[6,7], 5:[8,9], 6:[10,11], 7:[12,13], 8:[14,15],
            9:[16,17], 10:[18,19], 11:[20,21], 12:[22,23], 13:[24,25], 14:[26,27], 15:[28,29],
            16:[30,31], 17:[32,33], 18:[34,35] }
    pairs = [(18,1), (1,2), (1,3), (2,4), (3,5), #head
        (18,6), (18,7), (6,8), (7,9), (8,10), (9,11), #shouler,elbow, wrist
        (18,12), (18,13), (12,14), (13,15), (14,16),  (15,17) #legs
        ]

    DATASET_MIN_BBOX = torch.from_numpy(np.array(cfg.DATASET.MIN_BBOX)[None, None,:]).to('cuda')
    DATASET_MAX_BBOX = torch.from_numpy(np.array(cfg.DATASET.MAX_BBOX)[None, None,:]).to('cuda')
        
    
    K = pred_traj.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    # 1) Need to unnormalize and put back into absolute coordinate
    pred_traj = pred_traj*(DATASET_MAX_BBOX - DATASET_MIN_BBOX) + DATASET_MIN_BBOX
    target = target*(DATASET_MAX_BBOX - DATASET_MIN_BBOX) + DATASET_MIN_BBOX
    
    # 2) Bone Loss
    if cfg.DATASET.BONE:
        bone_loss = pred_traj - target
        bone_loss = (torch.abs(bone_loss) - DATASET_MIN_BBOX)/(DATASET_MAX_BBOX -DATASET_MIN_BBOX)
        bone_loss = torch.sum(bone_loss, dim=-1).mean() # might want to take difference first then normalize
    
    
    # Undo the relative  coordinates to get joints
    pred_traj_18 = pred_traj[:,:,:,kpoint[18][0]:kpoint[18][1]+1]
    target_18 =  target[:,:,:,kpoint[18][0]:kpoint[18][1]+1]
    # Right Face
    pred_traj_1 =  pred_traj[:,:,:,kpoint[1][0]:kpoint[1][1]+1] + pred_traj_18
    pred_traj_2 =  pred_traj[:,:,:,kpoint[2][0]:kpoint[2][1]+1] + pred_traj_1
    pred_traj_4 =  pred_traj[:,:,:,kpoint[4][0]:kpoint[4][1]+1] + pred_traj_2


    target_1 =  target[:,:,:,kpoint[1][0]:kpoint[1][1]+1] + target_18
    target_2 =  target[:,:,:,kpoint[2][0]:kpoint[2][1]+1] + target_1
    target_4 =  target[:,:,:,kpoint[4][0]:kpoint[4][1]+1] + target_2

    # Left Face
    pred_traj_3 =  pred_traj[:,:,:,kpoint[3][0]:kpoint[3][1]+1] + pred_traj_1
    pred_traj_5 =  pred_traj[:,:,:,kpoint[5][0]:kpoint[5][1]+1] + pred_traj_3


    target_3 =  target[:,:,:,kpoint[3][0]:kpoint[3][1]+1] + target_1
    target_5 =  target[:,:,:,kpoint[5][0]:kpoint[5][1]+1] + target_3

    #Right Arm
    pred_traj_6 =  pred_traj[:,:,:,kpoint[6][0]:kpoint[6][1]+1] + pred_traj_18
    pred_traj_8 =  pred_traj[:,:,:,kpoint[8][0]:kpoint[8][1]+1] + pred_traj_6
    pred_traj_10 =  pred_traj[:,:,:,kpoint[10][0]:kpoint[10][1]+1] + pred_traj_8


    target_6 =  target[:,:,:,kpoint[6][0]:kpoint[6][1]+1] + target_18
    target_8 =  target[:,:,:,kpoint[8][0]:kpoint[8][1]+1] + target_6
    target_10 =  target[:,:,:,kpoint[10][0]:kpoint[10][1]+1] + target_8

    # Left Arm
    pred_traj_7 =  pred_traj[:,:,:,kpoint[7][0]:kpoint[7][1]+1] + pred_traj_18
    pred_traj_9 =  pred_traj[:,:,:,kpoint[9][0]:kpoint[9][1]+1] + pred_traj_7
    pred_traj_11 =  pred_traj[:,:,:,kpoint[11][0]:kpoint[11][1]+1] + pred_traj_9


    target_7 =  target[:,:,:,kpoint[7][0]:kpoint[7][1]+1] + target_18
    target_9 =  target[:,:,:,kpoint[9][0]:kpoint[9][1]+1] + target_7
    target_11 =  target[:,:,:,kpoint[11][0]:kpoint[11][1]+1] + target_9

    # Right Leg
    pred_traj_12 =  pred_traj[:,:,:,kpoint[12][0]:kpoint[12][1]+1] + pred_traj_18
    pred_traj_14 =  pred_traj[:,:,:,kpoint[14][0]:kpoint[14][1]+1] + pred_traj_12
    pred_traj_16 =  pred_traj[:,:,:,kpoint[16][0]:kpoint[16][1]+1] + pred_traj_14

    target_12 =  target[:,:,:,kpoint[12][0]:kpoint[12][1]+1] + target_18
    target_14 =  target[:,:,:,kpoint[14][0]:kpoint[14][1]+1] + target_12
    target_16 =  target[:,:,:,kpoint[16][0]:kpoint[16][1]+1] + target_14

    # Left Leg
    pred_traj_13 =  pred_traj[:,:,:,kpoint[13][0]:kpoint[13][1]+1] + pred_traj_18
    pred_traj_15 =  pred_traj[:,:,:,kpoint[15][0]:kpoint[15][1]+1] + pred_traj_13
    pred_traj_17 =  pred_traj[:,:,:,kpoint[17][0]:kpoint[17][1]+1] + pred_traj_15

    target_13 =  target[:,:,:,kpoint[13][0]:kpoint[13][1]+1] + target_18
    target_15 =  target[:,:,:,kpoint[15][0]:kpoint[15][1]+1] + target_13
    target_17 =  target[:,:,:,kpoint[17][0]:kpoint[17][1]+1] + target_15
    
    

    data_pred_traj = torch.cat((pred_traj_1, pred_traj_2, pred_traj_3,
                                pred_traj_4, pred_traj_5, pred_traj_6,
                                pred_traj_7, pred_traj_8, pred_traj_9,
                                pred_traj_10, pred_traj_11, pred_traj_12,
                                pred_traj_13, pred_traj_14, pred_traj_15,
                                pred_traj_16, pred_traj_17,pred_traj_18), dim=3)

    data_target = torch.cat((target_1, target_2, target_3,
                            target_4, target_5, target_6,
                            target_7, target_8, target_9,
                            target_10, target_11, target_12,
                            target_13, target_14, target_15,
                            target_16, target_17, target_18), dim=3)


    # 3)  Joint Loss
    if cfg.DATASET.JOINT:
        joint_loss = data_pred_traj - data_target
        joint_loss = (torch.abs(joint_loss) - DATASET_MIN_BBOX)/(DATASET_MAX_BBOX -DATASET_MIN_BBOX)
        joint_loss = torch.sum(joint_loss, dim=-1).mean()

    

    # #4) Add the endpoint one I thought off
    #####################
    if cfg.DATASET.ENDPOINT:
        # joints = [2,4,3,5,8,10,9,11,14,16,15,17]
        # loss_endpoint = torch.tensor([0.0]).to('cuda')
        # for joint in joints:
        #     loss_diff_gt = (data_target[:,:,:,kpoint[joint][0]:kpoint[joint][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] )
        #     loss_diff_pred = (data_pred_traj[:,:,:,kpoint[joint][0]:kpoint[joint][1]+1] - data_pred_traj[:,:,:,kpoint[18][0]:kpoint[18][1]+1] )                   
        #     loss_endpoint = loss_endpoint + torch.abs(loss_diff_gt -loss_diff_pred)

        # endpoint_loss = (loss_endpoint- DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2]) 
        
        # endpoint_loss = endpoint_loss.mean()


        loss_right_arm = (data_pred_traj[:,:,:,kpoint[6][0]:kpoint[6][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] ) +\
                            (data_pred_traj[:,:,:,kpoint[8][0]:kpoint[8][1]+1] - data_target[:,:,:,kpoint[6][0]:kpoint[6][1]+1] ) +\
                            (data_pred_traj[:,:,:,kpoint[10][0]:kpoint[10][1]+1] - data_target[:,:,:,kpoint[8][0]:kpoint[8][1]+1] )

        loss_right_arm = (torch.abs(loss_right_arm) - DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2])

        #####################
        loss_left_arm = (data_pred_traj[:,:,:,kpoint[7][0]:kpoint[7][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[9][0]:kpoint[9][1]+1] - data_target[:,:,:,kpoint[7][0]:kpoint[7][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[11][0]:kpoint[11][1]+1] - data_target[:,:,:,kpoint[9][0]:kpoint[9][1]+1] )

        loss_left_arm = (torch.abs(loss_left_arm) - DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2])
        #################
        loss_left_leg = (data_pred_traj[:,:,:,kpoint[13][0]:kpoint[13][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[15][0]:kpoint[15][1]+1] - data_target[:,:,:,kpoint[13][0]:kpoint[13][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[17][0]:kpoint[17][1]+1] - data_target[:,:,:,kpoint[15][0]:kpoint[15][1]+1] )
        
        loss_left_leg = (torch.abs(loss_left_leg) - DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2])
        #################
        
        loss_right_leg = (data_pred_traj[:,:,:,kpoint[12][0]:kpoint[12][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[14][0]:kpoint[14][1]+1] - data_target[:,:,:,kpoint[12][0]:kpoint[12][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[16][0]:kpoint[16][1]+1] - data_target[:,:,:,kpoint[14][0]:kpoint[14][1]+1] )

        loss_right_leg = (torch.abs(loss_right_leg) - DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2])
        #################

        loss_left_face = (data_pred_traj[:,:,:,kpoint[1][0]:kpoint[1][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[3][0]:kpoint[3][1]+1] - data_target[:,:,:,kpoint[1][0]:kpoint[1][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[5][0]:kpoint[5][1]+1] - data_target[:,:,:,kpoint[3][0]:kpoint[3][1]+1] )

        loss_left_face = (torch.abs(loss_left_face) - DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2])
        #################

        loss_right_face = (data_pred_traj[:,:,:,kpoint[1][0]:kpoint[1][1]+1] - data_target[:,:,:,kpoint[18][0]:kpoint[18][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[2][0]:kpoint[2][1]+1] - data_target[:,:,:,kpoint[1][0]:kpoint[1][1]+1] ) +\
                        (data_pred_traj[:,:,:,kpoint[4][0]:kpoint[4][1]+1] - data_target[:,:,:,kpoint[2][0]:kpoint[2][1]+1] )

        loss_right_face = (torch.abs(loss_right_face) - DATASET_MIN_BBOX[:,:,:2])/(DATASET_MAX_BBOX[:,:,:2] - DATASET_MIN_BBOX[:,:,:2])
        #################
        endpoint_loss = loss_right_arm.mean() + loss_left_arm.mean() + loss_left_leg.mean() +loss_right_leg.mean() + loss_left_face.mean() + loss_right_face.mean()

    if not cfg.DATASET.ENDPOINT and not cfg.DATASET.BONE and not cfg.DATASET.JOINT:
        return torch.tensor([0.0]).to('cuda'), torch.tensor([0.0]).to('cuda'), torch.tensor([0.0]).to('cuda') #joint_loss
    elif cfg.DATASET.ENDPOINT and cfg.DATASET.BONE and cfg.DATASET.JOINT:
        return joint_loss, bone_loss, endpoint_loss
    elif cfg.DATASET.JOINT and cfg.DATASET.ENDPOINT:
        return joint_loss, torch.tensor([0.0]).to('cuda'), endpoint_loss #joint_loss and endpoint_loss
    elif cfg.DATASET.JOINT and cfg.DATASET.BONE:
        return joint_loss, bone_loss, torch.tensor([0.0]).to('cuda') #bone and joint_loss
    elif cfg.DATASET.BONE  and cfg.DATASET.ENDPOINT:
        return torch.tensor([0.0]).to('cuda'), bone_loss, endpoint_loss #bone and endpoint_loss
    elif cfg.DATASET.BONE:
        return torch.tensor([0.0]).to('cuda'), bone_loss, torch.tensor([0.0]).to('cuda') #bone loss
    elif cfg.DATASET.ENDPOINT:
        return torch.tensor([0.0]).to('cuda'), torch.tensor([0.0]).to('cuda'), cfg.DATASET.SCALE_END_FACTOR*endpoint_loss
    elif cfg.DATASET.JOINT:
        return joint_loss, torch.tensor([0.0]).to('cuda'), torch.tensor([0.0]).to('cuda') #joint_loss

    # return joint_loss, torch.tensor([0.0]).to('cuda'), torch.tensor([0.0]).to('cuda')


def bom_traj_loss(pred, target):
    '''
    pred: (B, T, K, dim)
    target: (B, T, dim)
    '''
    K = pred.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    return loss_traj

def fol_rmse(x_true, x_pred):
    '''
    Params:
        x_pred: (batch, T, pred_dim) or (batch, T, K, pred_dim)
        x_true: (batch, T, pred_dim) or (batch, T, K, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''

    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=-1))#
    L2_diff = torch.sum(L2_diff, dim=-1).mean()

    return L2_diff