#python tools/test.py --config_file configs/bitrap_np_JAAD.yml CKPT_DIR /home/akanu/data/JAAD/checkpoints/bitrap_np_K_20.pth 
# python tools/test.py --config_file configs/avenue.yml CKPT_DIR /home/akanu/checkpoints/JAAD_checkpoints/goal_cvae_checkpoints/no_wandb/avenue_input_5_ouput_5.pth

# python tools/test.py --config_file configs/avenue.yml CKPT_DIR /home/akanu/checkpoints/JAAD_checkpoints/goal_cvae_checkpoints/5_input_10_pred/no_wandb/avenue_input_5_output_10.pth

# python tools/test.py --config_file configs/avenue.yml CKPT_DIR /home/akanu/checkpoints/JAAD_checkpoints/goal_cvae_checkpoints/5_input_10_pred/no_wandb/avenue_input_5_output_10.pth


# avenue
# python tools/test.py --config_file configs/avenue.yml CKPT_DIR /home/akanu/checkpoints/JAAD_checkpoints/goal_cvae_checkpoints/wandb_unimodal/avenue_input_25_output_25.pth

# python tools/test.py --config_file configs/avenue_pose.yml CKPT_DIR /home/akanu/checkpoints/JAAD_checkpoints/goal_cvae_checkpoints/wandb_unimodal_pose/gua_in_13_out_13_k_1.pth


python tools/test.py --config_file configs/avenue_pose_human_constraint.yml CKPT_DIR /home/akanu/checkpoints/JAAD_checkpoints/goal_cvae_checkpoints/wandb_unimodal_pose_human_constraint/gua_in_25_out_25_k_1_hc_no_bone_endpoint_joint.pkl
