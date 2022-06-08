# #4) Add the endpoint one I thought off
    #####################
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
