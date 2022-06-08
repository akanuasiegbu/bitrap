import numpy as np
import os
from PIL import Image
import torch
from torch.utils import data

from load_pose_traj import load_poses
from config_for_my_data import hyparams, loc, exp

class PoseLoader(data.Dataset):
    def __init__(self, cfg, split, train_poses, test_poses):
        self.split = split
        # self.root = cfg.DATASET.ROOT
        self.cfg = cfg
        

        if cfg.DATASET.NAME_SECOND =='st':
            # to extract ##_#### of st dataset 
            if self.split == 'train' or self.split == 'val':
                file_num = os.listdir(cfg.DATASET.ST_VID_DIR_TRAIN)
            else:
                file_num = os.listdir(cfg.DATASET.ST_VID_DIR_TEST)

            file_num.sort()
            vids = []
            for i in file_num:
                split =i.split('_')
                vids.append([int(split[0]), int(split[1][:-4])])
        
        elif cfg.DATASET.NAME_SECOND =='avenue':
            if self.split =='train' or self.split == 'val':
                vids=range(1,17)
            else:
                vids=range(1,22)

        # Need to change this below to load generally
        if self.split == 'train' or self.split == 'val':
            # traj_loc_format = '/mnt/roahm/users/akanu/projects/AlphaPose/avenue_alphapose_out/train/{}/alphapose-results.json'
            self.data = load_poses(train_poses, vids, input_seq=hyparams['input_seq'], pred_seq=hyparams['pred_seq'], dataset=cfg.DATASET.NAME_SECOND, train_or_test ='train')
        else:                
            assert self.split == 'test'
            # traj_loc_format = '/mnt/roahm/users/akanu/projects/anomalous_pred/output_pose_json_appended/{:02d}_append.json'
            self.data = load_poses(test_poses, vids, input_seq=hyparams['input_seq'], pred_seq=hyparams['pred_seq'], dataset=cfg.DATASET.NAME_SECOND)
            self.global_pose_to_relative_pose()
            self.x = self.convert_normalize_bboxes(self.data['x'])
            self.y = self.convert_normalize_bboxes(self.data['y'])

       
        if self.split == 'train':
            np.random.seed(49)
            rand = np.random.permutation(len(self.data['x']))
            step = int(len(self.data['x'])*0.7)
            train = {}
            for key in self.data:
                train[key] = self.data[key][rand][:step]
            
            self.data = train
            self.global_pose_to_relative_pose()
            self.x = self.convert_normalize_bboxes(self.data['x'])
            self.y = self.convert_normalize_bboxes(self.data['y'])
        
        elif self.split == 'val':
            np.random.seed(49)
            rand = np.random.permutation(len(self.data['x']))
            step = int(len(self.data['x'])*0.7)

            val = {}
            for key in self.data:
                val[key] = self.data[key][rand][step:] # to all for the correct batch

            self.data = val
            self.global_pose_to_relative_pose()
            self.x = self.convert_normalize_bboxes(self.data['x'])
            self.y = self.convert_normalize_bboxes(self.data['y'])

        # Might need to check on it
        # self.global_pose_to_relative_pose_from_fixed_point(

    def __getitem__(self, index):
        obs_pose = torch.FloatTensor(self.x[index])
        pred_pose = torch.FloatTensor(self.y[index])
        id_x = torch.from_numpy(self.data['id_x'][index])
        id_y = torch.from_numpy(np.array(self.data['id_y'][index]))
        frame_x = torch.from_numpy(self.data['frame_x'][index])
        frame_y = torch.from_numpy(np.array(self.data['frame_y'][index]))
        vid_name = torch.from_numpy(np.array(self.data['video_file'][index]))

        kp_confidence_pred = torch.from_numpy(np.array(self.data['kp_confidence_pred'][index]))
        abnormal_ped_input = torch.from_numpy(np.array(self.data['abnormal_ped_input'][index]))
        abnormal_ped_pred = torch.from_numpy(np.array(self.data['abnormal_ped_pred'][index]))
        abnormal_gt_frame = torch.from_numpy(np.array(self.data['abnormal_gt_frame'][index]))

        
        ret = {'input_x':obs_pose, 'target_y':pred_pose, 'video_file':vid_name,
                'id_x':id_x, 'id_y':id_y, 'frame_x':frame_x, 'frame_y':frame_y,
                'kp_confidence_pred':kp_confidence_pred,
                'abnormal_ped_input':abnormal_ped_input, 'abnormal_ped_pred':abnormal_ped_pred,
                'abnormal_gt_frame':abnormal_gt_frame}
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    
    def convert_normalize_bboxes(self, all_bboxes):
        '''input box type is x1y1x2y2 in original resolution'''
        _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
        _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]

        # _max = np.array([640, 360])[None,:]
        # _min = np.zeros((1,38))
        # _max = np.tile(_max,19)

        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = all_bboxes[i]
            bbox = (bbox - _min) / (_max - _min)
            # NOTE ltrb to cxcywh

            # W, H  = all_resolutions[i][0]
            #     bbox = (bbox - _min) / (_max - _min)
            # if self.cfg.DATASET.NORMALIZE == 'zero-one':
            # elif self.cfg.DATASET.NORMALIZE == 'plus-minus-one':
            #     # W, H  = all_resolutions[i][0]
            #     bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            # elif self.cfg.DATASET.NORMALIZE == 'none':
            #     pass
            # else:
            #     raise ValueError(self.cfg.DATASET.NORMALIZE)

            all_bboxes[i] = bbox
        return all_bboxes
  
    # def estimated_center_point_from_pose(self, data):
    #     max_x = np.max(data[:,:,::2], axis=2)
    #     min_x = np.min(data[:,:,::2], axis=2)
    #     max_y = np.max(data[:,:,1::2], axis=2)
    #     min_y = np.min(data[:,:,1::2], axis=2)
    #     center_point_x = (max_x+ min_x) / 2
    #     center_point_y = (max_y+ min_y) / 2

    #     return center_point_x, center_point_y


    def estimated_center_point_from_pose(self, data):
        left_shoulder_x = data[:,:,12]
        left_shoulder_y = data[:,:,13]
        
        right_shoulder_x = data[:,:,10]
        right_shoulder_y = data[:,:,11]
        
        left_hip_x = data[:,:,24]
        left_hip_y = data[:,:,25]

        right_hip_x = data[:,:,22]
        right_hip_y = data[:,:,23]

        center_point_x = (left_shoulder_x+ right_shoulder_x + left_hip_x + right_hip_x) / 4
        center_point_y = (left_shoulder_y+ right_shoulder_y + left_hip_y + right_hip_y) / 4

        return center_point_x, center_point_y

    def relative_difference_from_fixed_point(self, data):
        center_point_x, center_point_y = data[:,:, 34:35], data[:,:, 35:36]
        data_rel_for_x = data - center_point_x
        data_rel_for_y = data - center_point_y
        data_rel_for_x[:,:, 1:36:2] = 0
        data_rel_for_y[:,:, :35:2] = 0

        data = data_rel_for_x + data_rel_for_y
        data = np.concatenate((data[:,:,:34], center_point_x, center_point_y), axis=2)
        return data

    def relative_difference(self, data):
        kpoint = {1: [0,1], 2:[2,3], 3:[4,5], 4:[6,7], 5:[8,9], 6:[10,11], 7:[12,13], 8:[14,15],
                    9:[16,17], 10:[18,19], 11:[20,21], 12:[22,23], 13:[24,25], 14:[26,27], 15:[28,29],
                    16:[30,31], 17:[32,33], 18:[34,35] }
        pairs = [(18,1), (1,2), (1,3), (2,4), (3,5), #head
                (18,6), (18,7), (6,8), (7,9), (8,10), (9,11), #shouler,elbow, wrist
                (18,12), (18,13), (12,14), (13,15), (14,16),  (15,17) #legs
                ]
        
        temp ={}
        
        for pair in pairs:
            end = pair[1]
            start = pair[0]
            temp[end] = data[:,:, kpoint[end][0]: kpoint[end][1] + 1 ] - data[:,:, kpoint[start][0]: kpoint[start][1] + 1 ]
        
        temp[18] = data[:,:,-2:] #root joint
        indices = np.arange(2,19)
        
        data_rel = temp[1]
        for index in indices:
            data_rel = np.concatenate((data_rel, temp[index]), axis=2)

        return data_rel



    def global_pose_to_relative_pose_from_fixed_point(self):
    
        # do I first normalize the poses? idk
        # Lets forget normaliztion rn and just think how to make relative to a fixed point
        # Need to make sure input is only poses
        if self.data['x'].shape[2] == 38:
            self.data['x'] = self.data['x'][:,:,:-4] #removes appended bounding boxes
            self.data['y'] = self.data['y'][:,:,:-4] #removes appended bounding boxes

        input_center_point_x, input_center_point_y = self.estimated_center_point_from_pose(self.data['x'])
        output_center_point_x, output_center_point_y = self.estimated_center_point_from_pose(self.data['y'])

        input_center_point_x_and_y = np.concatenate((np.expand_dims(input_center_point_x, axis=2), np.expand_dims(input_center_point_y, axis=2)), axis=2)
        output_center_point_x_and_y = np.concatenate((np.expand_dims(output_center_point_x, axis=2), np.expand_dims(output_center_point_y, axis=2)), axis=2)
        self.data['x'] = np.concatenate((self.data['x'], input_center_point_x_and_y), axis =2)
        self.data['y'] = np.concatenate((self.data['y'], output_center_point_x_and_y), axis=2)
        
        self.data['x'] = self.relative_difference_from_fixed_point(self.data['x'])
        self.data['y'] = self.relative_difference_from_fixed_point(self.data['y'])

    def global_pose_to_relative_pose(self):
        if self.data['x'].shape[2] == 38:
            self.data['x'] = self.data['x'] [:,:,:-4] #removes appended bounding boxes
            self.data['y'] = self.data['y'][:,:,:-4] #removes appended bounding boxes

        input_center_point_x, input_center_point_y = self.estimated_center_point_from_pose(self.data['x'])
        output_center_point_x, output_center_point_y = self.estimated_center_point_from_pose(self.data['y'])

        input_center_point_x_and_y = np.concatenate((np.expand_dims(input_center_point_x, axis=2), np.expand_dims(input_center_point_y, axis=2)), axis=2)
        output_center_point_x_and_y = np.concatenate((np.expand_dims(output_center_point_x, axis=2), np.expand_dims(output_center_point_y, axis=2)), axis=2)
        self.data['x'] = np.concatenate((self.data['x'], input_center_point_x_and_y), axis =2)
        self.data['y'] = np.concatenate((self.data['y'], output_center_point_x_and_y), axis=2)

        self.data['x'] = self.relative_difference(self.data['x'])
        self.data['y'] = self.relative_difference(self.data['y'])
        

if __name__ == "__main__":
    train_poses =  loc['data_load']['st']['train_poses']
    test_poses =  loc['data_load']['st']['test_poses']

    poses_train= PoseLoader('cfg','train', train_poses, test_poses)
    print('done val')

    i = 0
    for train in zip(poses_train):
        
        i +=1
        if i == 3:
            break
        print('*'*20)
    quit()
    

    # for data 
    print('done')