import json
import numpy as np
from scipy.io import loadmat
import os
from config_for_my_data import loc

def load_poses(traj_loc_format, vids, input_seq, pred_seq, dataset, train_or_test ='test', load_type='poses_bbox', data_consecutive = True, window = 1):
    #Load a single video
    # Load poses and bounding box
    #Load poses
    # Load bounding box
    
    x, y, frame_x, frame_y, id_x, id_y, vid_name = [], [], [], [], [], [], []
    abnormal_ped_input, abnormal_ped_pred = [] , []
    abnormal_gt = []
    kp_confidence_pred = []
    for vid in vids:
        if dataset == 'avenue':
            f = open(traj_loc_format.format(vid))
        elif dataset =='st':
            f = open(traj_loc_format.format(vid[0], vid[1]))
        datas= json.load(f)
        datas = np.array(datas) 
        
        # Extracts image_id('frame number') and idx
        # Saves index
        # Note ending with vid implies video data
        index_vid, image_id_vid, idx_vid = [], [], []
        for i, data in enumerate(datas):
            index_vid.append(i)
            image_id_vid.append(data['image_id'])
            idx_vid.append(data['idx'])
            
        
        # ids is a matrix with columns image_id, idx and index    
        ids = np.array([index_vid, image_id_vid, idx_vid]).T
        
        index_vid = np.array(index_vid).T
        image_id_vid = np.array(image_id_vid).T
        idx_vid = np.array(idx_vid).T
        
        max_idx = ids[:,2].max()
        
        
        for num in range(1,max_idx+1):
            # Need to select idx
            # Then select index corresponding to idx
            # Then build trajectory
            
            # Ending with ped implies looking at pedestrain
            locs_ped = np.where(idx_vid == num)[0]
            seq_len_ped = len(locs_ped)
            
            image_id_ped = image_id_vid[locs_ped]
            idx_ped = idx_vid[locs_ped]
            
            # Subset list containing ped idx
            temp_ped = datas[locs_ped]
                        
            if seq_len_ped >= (input_seq + pred_seq):
                for i in range(0, seq_len_ped - input_seq - pred_seq + 1, window):
                    
                    start_input = i
                    end_input = i + input_seq
                    end_output = i + input_seq + pred_seq
                    x_and_y = image_id_ped[start_input:end_output]

                    if data_consecutive:
                        # This ensures that data inputted is from consecutive frames 
                        first_frame_input = image_id_ped[start_input]
                        last_frame_output = image_id_ped[start_input] + input_seq + pred_seq
                        check_seq = np.cumsum(x_and_y) == np.cumsum(np.arange(first_frame_input, last_frame_output, 1))
                        if not np.all(check_seq):
                            continue
                        
                    #  x, y, frame_x, frame_y, id_x, id_y, vid_name
                    # Need to go inside of the temp_ped to extract data  
                    
                    # Enable option to save poses and bouding boxes appended to end
                    temp_x, temp_frame_x, temp_id_x = [], [], []
                    temp_y, temp_frame_y, temp_id_y = [], [], []
                    temp_abnormal_ped_input, temp_abnormal_ped_pred = [] , []
                    temp_abnormal_gt = []
                    temp_kp_confidence_pred = []
                    for temp in  temp_ped[start_input:end_input]:
                        # bbox default save to json file from AlphaPose is in tlwh
                        # change it to mid_x,mid_y, w, h
                        bbox = temp['box']
                        bbox = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]] # xywh
                        traj = np.array(temp['keypoints']).reshape(17,3)
                        
                        # bbox only
                        if load_type == 'bbox':# poses + bbox
                            temp_x.append(np.array(bbox))
                        # poses only
                        elif load_type == 'poses':
                            temp_x.append(traj[:,:2].reshape(-1))
                        elif load_type == 'poses_bbox':
                            temp_x.append(np.hstack((traj[:,:2].reshape(-1), np.array(bbox))))
                        
                        temp_id_x.append(temp['idx'])
                        temp_frame_x.append(temp['image_id'])
                        if train_or_test =='test':
                            temp_abnormal_ped_input.append(temp['abnormal_pedestrain']) 
                        else:
                            temp_abnormal_ped_input.append(0)
                    for temp in temp_ped[end_input:end_output]:
                        # bbox default save to json file from AlphaPose is in tlwh
                        # change it to mid_x,mid_y, w, h
                        bbox = temp['box']
                        bbox = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]] # xywh
                        traj = np.array(temp['keypoints']).reshape(17,3)
                        
                        # bbox only
                        if load_type == 'bbox':# poses + bbox
                            temp_y.append(np.array(bbox))
                        # poses only
                        elif load_type == 'poses':
                            temp_y.append(traj[:,:2].reshape(-1))
                        elif load_type == 'poses_bbox':
                            temp_y.append(np.hstack((traj[:,:2].reshape(-1), np.array(bbox))))

                        temp_kp_confidence_pred.append(traj[:,2].reshape(-1))

                        temp_id_y.append(temp['idx'])
                        temp_frame_y.append(temp['image_id'])
                        if train_or_test == 'test':
                            temp_abnormal_ped_pred.append(temp['abnormal_pedestrain'])
                            temp_abnormal_gt.append(temp['abnormal_gt_frame'])
                        else: 
                            temp_abnormal_ped_pred.append(0)
                            temp_abnormal_gt.append(0)
                    # Appending each tracjectory 
                    # So one index corresponds to trajectory
                    x.append(temp_x)
                    id_x.append(temp_id_x)
                    frame_x.append(temp_frame_x)
                    abnormal_ped_input.append(temp_abnormal_ped_input)    
                    
                    y.append(temp_y)
                    id_y.append(temp_id_y)
                    frame_y.append(temp_frame_y)
                    abnormal_ped_pred.append(temp_abnormal_ped_pred)
                    kp_confidence_pred.append(temp_kp_confidence_pred)
                    
                    abnormal_gt.append(temp_abnormal_gt)
                    vid_name.append(vid)
                    
            else:
                continue
        
            # for i in range
        
    output ={}
    output['x'] = np.array(x)
    output['id_x'] = np.array(id_x)
    output['frame_x'] = np.array(frame_x)
    output['abnormal_ped_input'] = np.array(abnormal_ped_input, dtype=np.int8)
    
    output['y'] = np.array(y)
    output['id_y'] = np.array(id_y)
    output['frame_y'] = np.array(frame_y)
    output['abnormal_ped_pred'] = np.array(abnormal_ped_pred, dtype=np.int8)
    output['kp_confidence_pred'] = np.array(kp_confidence_pred)

    # Note that only kept the abnormal_gt_frame of traj to be predicted    
    output['abnormal_gt_frame'] = np.array(abnormal_gt, dtype=np.int8)
    output['video_file'] = np.array(vid_name)
    
    return output


    
if __name__ == '__main__':
    
    # traj_loc_format = '/mnt/roahm/users/akanu/projects/AlphaPose/avenue_alphapose_out/train/{}/alphapose-results.json'
    # vids=range(1,17)
    # poses = load_poses(traj_loc_format, vids , input_seq=3, pred_seq=3, train_or_test='train')
    # print('done')
    


    file_num = os.listdir('/mnt/roahm/users/akanu/dataset/Anomaly/ShangaiuTech/testing/videos_from_frame')
    # file_num = os.listdir('/mnt/workspace/datasets/shanghaitech/training/videos')
    file_num.sort()
    vids = []
    for i in file_num:
        split =i.split('_')
        vids.append([int(split[0]), int(split[1][:-4])])


    train_poses =  loc['data_load']['st']['train_poses']
    test_poses =  loc['data_load']['st']['test_poses']
    poses = load_poses(test_poses, vids[:1] , input_seq=3, pred_seq=3, dataset='st', train_or_test='test')
    print('done')
    
    
    