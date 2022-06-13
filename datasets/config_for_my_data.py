import datetime

exp = { 
        'data': 'avenue', #st, avenue
        'data_consecutive': True,
        }


hyparams = {
    'batch_size': 32,
    'buffer_size': 10000,
    
    'input_seq': 25,
    'pred_seq': 25,

    'to_xywh':True, # This is assuming file is in tlbr format

}




loc =  {
    
    'nc':{
        'data_coordinate_out': 'xywh',
        'dataset_name': exp['data'], # avenue, st             
        },   

    'data_load':
            'avenue':{
                'train_file': "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/train_txt/", # bbox
                'test_file': "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/test_txt/", #bbox
                'train_vid': '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/training_videos',
                'test_vid': '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/testing_videos',
                'train_poses': '/mnt/roahm/users/akanu/projects/AlphaPose/avenue_alphapose_out/train/{}/alphapose-results.json', #poses
                'test_poses': '/mnt/roahm/users/akanu/projects/anomalous_pred/output_pose_json_appended/{:02d}_append.json', #poses
                },

            'st':{
                'train_file':"/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/train_txt/", #bbox
                "test_file": "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/test_txt/", #bbox
                'train_vid': '/mnt/workspace/datasets/shanghaitech/training/videos',
                'test_vid':  '/mnt/roahm/users/akanu/projects/Deep-SORT-YOLOv4/tensorflow2.0/deep-sort-yolov4/input_video/st_test',
                'train_poses': '/mnt/roahm/users/akanu/projects/AlphaPose/st_alphapose_out/train/{:02d}_{:03d}.avi/alphapose-results.json', #poses
                'test_poses': '/mnt/roahm/users/akanu/projects/anomalous_pred/output_pose_json_appended_st/{:02d}_{:04d}_append.json', #poses
                },
            'corridor':{
                'train_file': '/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/corridor/train_txt/',
                'test_file': '/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/corridor/test_txt/',
                # 'train_vid':
                # 'test_vid':
            },
            }
            


}
