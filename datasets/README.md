# 文件与数据集说明

## 数据集

| DataSet | Model | Status    | FileName                      | DataSetKeys                                                                                                                                                                                       |
|---------|-------|-----------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AMASS   | smplx | ADAPTED   | C4 Run to walk1poses.npz      | trans, gender, mocap_framerate, betas, dmpls, poses                                                                                                                                               |
|         |       |           | G18 push kick right poses.npz |                                                                                                                                                                                                   |
| HuMMan  | smpl  | UNADAPTED | p000528_a001026.npz           | transl, body_pose, global_orient, betas                                                                                                                                                           |
|         |       |           | p000552_a099999.npz           |                                                                                                                                                                                                   |
| 3DPW    | smpl  | UNADAPTED | courtyard_basketball_00.pkl   | trans_60Hz, cam_intrinsics, poses, img_frame_ids, betas_clothed, sequence, v_template_clothed, jointPositions, poses_60Hz, betas, cam_poses, campose_valid, genders, trans, poses2d, texture_maps |
|         |       |           | office_phoneCall_00.pkl       |                                                                                                                                                                                                   |

对于 3DPW 数据集，读取方式为：

```python
import pickle as pkl

DATASET_3DPW = r'../datasets/3DPW/office_phoneCall_00.pkl'

data = pkl.load(open(DATASET_3DPW, 'rb'), encoding='bytes')
print(', '.join(k.decode() for k in data.keys()))
```

## 模型

| Model | ModelKeys                                                                                                                                                                                                                                                                                        |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| smplx | hands_meanr, hands_meanl, lmk_bary_coords, vt, posedirs, part2num, hands_coeffsr, lmk_faces_idx, J_regressor, dynamic_lmk_faces_idx, hands_componentsr, shapedirs, dynamic_lmk_bary_coords, ft, hands_componentsl, joint2num, v_template, allow_pickle, f, hands_coeffsl, kintree_table, weights |
| smpl  | J_regressor_prior, f, J_regressor, kintree_table, J, weights_prior, weights, posedirs, v_template, shapedirs                                                                                                                                                                                     |

