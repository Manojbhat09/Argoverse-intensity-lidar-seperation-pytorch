# Converting the ply files into the bin files

import sys 

import pyntcloud
import numpy as np
import glob
import pdb
import os
import json
import matplotlib.pyplot as plt
import argoverse
import shutil
from tqdm import tqdm
import copy
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import argoverse.visualization.visualization_utils as viz_util
from scipy.spatial.transform import Rotation
import object3d
from functools import reduce

low_pointed=0
total_saved = 0
total_objects = 0
GROUND_REMOVED_PLANE = 0

object_dict = {
    'PEDESTRIAN': 3,
    'BICYCLE' : 2,
    'VEHICLE':1
}

# argo_to_kitti = np.array([[6.927964e-03, -9.999722e-01, -2.757829e-03],
#                                [-1.162982e-03, 2.749836e-03, -9.999955e-01],
#                                [9.999753e-01, 6.931141e-03, -1.143899e-03]]) 

argo_to_kitti = np.array([[0, -1, 0],
                           [0, 0, -1],
                           [1, 0, 0 ]])
vlp32_planes = {31: -25.,
                30: -15.639,
                29: -11.31,
                28: -8.843,
                27: -7.254,
                26: -6.148,
                25: -5.333,
                24: -4.667,
                23: -4.,
                22: -3.667,
                21: -3.333,
                20: -3.,
                19: -2.667,
                18: -2.333,
                17: -2.,
                16: -1.667,
                15: -1.333,
                14: -1.,
                13: -0.667,
                12: -0.333,
                11: 0.,
                10: 0.333,
                9:  0.667,
                8:  1.,
                7:  1.333,
                6:  1.667,
                5:  2.333,
                4:  3.333,
                3:  4.667,
                2:  7.,
                1:  10.333,
                0:  15.}

tf_down_lidar_rot = Rotation.from_quat([-0.9940207559208627, -0.10919018413803058, -0.00041138986312043766, -0.00026691721622102603])
tf_down_lidar_tr = [1.3533224859271054, -0.0009818949950377448, 1.4830535977952262]

tf_down_lidar = np.eye(4)
tf_down_lidar[0:3,0:3] = tf_down_lidar_rot.as_dcm()
tf_down_lidar[0:3,3] = tf_down_lidar_tr

tf_up_lidar_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
tf_up_lidar_tr = [1.35018, 0.0, 1.59042]
tf_up_lidar = np.eye(4)
tf_up_lidar[0:3,0:3] = tf_up_lidar_rot.as_dcm()
tf_up_lidar[0:3,3] = tf_up_lidar_tr

print('UP: ', tf_up_lidar_tr, tf_up_lidar_rot.as_rotvec(), tf_up_lidar_rot.as_euler('zyx', degrees=True), tf_up_lidar_rot.as_euler('xyz', degrees=True))
print('DOWN: ', tf_down_lidar_tr, tf_down_lidar_rot.as_rotvec(), tf_down_lidar_rot.as_euler('zyx', degrees=False), tf_down_lidar_rot.as_euler('xyz', degrees=False))


def ground_segmentation(pc,iter_cycle = 20, threshold = 0.12):
    
    pc = valid_region(pc,{'x':[-60,60],'y':[-40,40],'z':[-3.5,3.5]})
    pc = np.hstack((pc,np.arange(pc.shape[0],dtype=int).reshape(-1,1)))
    pc_orig = copy.deepcopy(pc)
    
    pc = valid_region(pc,{'x':[-60,60],'y':[-40,40],'z':[-3,0]})
    h_col = np.argmin(np.var(pc[:,:3], axis=0))

    bins, z_range = np.histogram(pc[:,h_col],20)
    approx_z = z_range[np.argmax(bins)]
    for n in range(iter_cycle):
        cov_mat = np.cov(pc[:,:3].T)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        
        normal_vector = eig_vec[np.argmin(eig_val)]
        height = np.dot(pc[:,:3],normal_vector)-np.dot(np.array([0,0,approx_z]),normal_vector)
        threshold_mask = np.abs(height) < threshold
        pc = pc[threshold_mask]

    world_mask = np.invert(np.in1d(pc_orig[:,3],pc[:,3]))
    world_points = pc_orig[:,:3][world_mask]
    ground_points = pc[:,:3]

    return np.array(world_points,dtype=np.float32), world_mask

def valid_region(pc,constraint):

    mask = ((pc[:,0] >= constraint['x'][0]) & (pc[:,0] <= constraint['x'][1])) \
           & ((pc[:,1] >= constraint['y'][0]) & (pc[:,1] <= constraint['y'][1]))\
           & ((pc[:,2] >= constraint['z'][0]) & (pc[:,2] <= constraint['z'][1])
             )

    valid_world_points = pc[mask]
    return valid_world_points
    
    
def check_make_(save_dir, display=True):
    if not os.path.exists(save_dir):
        if display:
            print("Making folder ", save_dir)
        os.makedirs(save_dir)
    else:
        if display:
            print("Folder present ", save_dir)
        else:
            pass

def dump_json(calib_file, calib):
    with open(calib_file, 'w') as outfile:
        json.dump(calib, outfile, ensure_ascii=False, indent=4)
        
def dump_txt(calib_file, calib):
    np.savetxt(calib_file, calib, delimiter=',')
    
def dump_pickle(calib_file, calib):
    np.save(calib_file, calib)

def dump_img(file, img):
    plt.imsave(file, img , format='png')
        
def dump_pcd(file, pcd):
    pcd.tofile(file)


def get_objects_from_label(label_file):
    # Opens a label file, and passes the object to Object3d object, Read the json GT labels
    f = open(label_file)
    label_data = json.load(f) 
    objects = [object3d.Object3d(data) for data in label_data]
    return objects
    
    
def homogeneous_transformation(points, translation, theta):
    return (points[:, :3] - translation).dot(rotation_matrix_3D(theta).T)

def rotation_matrix_3D(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])

    
def filter_pointcloud(bbox, pointcloud):
    theta = bbox.ry #["angle"]
    transformed_pointcloud = homogeneous_transformation(pointcloud[:, :3], bbox.pos[:3], theta)#["center"]
    if bbox.l > bbox.w:
        length = bbox.l
        width = bbox.w
    else:
        length = bbox.w
        width = bbox.l
    indices = reduce(np.intersect1d, (np.where(np.abs(transformed_pointcloud[:,0]) <= width/2)[0], 
                            np.where(np.abs(transformed_pointcloud[:,1]) <= length/2)[0],
                            np.where(np.abs(transformed_pointcloud[:,2]) <= bbox.h/2)[0]))
    return indices, transformed_pointcloud[indices,:]

def separate_pc(pc, tf_up, tf_down):

    pc_points = np.ones((len(pc), 4))
    pc_points[:,0:3] = pc[:,0:3]

    pc_up_tf = np.dot(np.linalg.inv(tf_up),  pc_points.transpose())
    pc_down_tf = np.dot(np.linalg.inv(tf_down), pc_points.transpose())

    pc_up_dis = np.sqrt(pc_up_tf[0,:]**2 + pc_up_tf[1,:]**2 + pc_up_tf[2,:]**2)
    pc_up_omega = np.arcsin(pc_up_tf[2,:]/pc_up_dis) * 180 / np.pi

    pc_down_dis = np.sqrt(pc_down_tf[0,:]**2 + pc_down_tf[1,:]**2 + pc_down_tf[2,:]**2)
    pc_down_omega = np.arcsin(pc_down_tf[2,:]/pc_down_dis) * 180 / np.pi

    pc_angles = np.array([vlp32_planes[pc[i,4]] for i in range(0, len(pc))])

    pc_up_xyz = pc_up_tf[:, (np.fabs(pc_up_omega - pc_angles) < 2.0) * (np.fabs(pc_down_omega - pc_angles) > 2.0)]
    pc_down_xyz = pc_down_tf[:, (np.fabs(pc_up_omega - pc_angles) > 2.0) * (np.fabs(pc_down_omega -  pc_angles) < 2.0)]

    pc_up = np.zeros((pc_up_xyz.shape[1], 5))
    pc_down = np.zeros((pc_down_xyz.shape[1], 5))

    pc_up[:,0:3] = pc_up_xyz.transpose()[:,0:3]
    pc_down[:,0:3] = pc_down_xyz.transpose()[:,0:3]
    pc_up[:,3:5] = pc[(np.fabs(pc_up_omega - pc_angles) < 2.0) * (np.fabs(pc_down_omega - pc_angles) > 2.0), 3:5]
    pc_down[:,3:5] = pc[(np.fabs(pc_up_omega - pc_angles) > 2.0) * (np.fabs(pc_down_omega -  pc_angles) < 2.0), 3:5]


    return pc_up, pc_down
    
def process_pcd(pts_lidar, label_file, pcd_folder, ind_folder, log_idx, pcd_idx, map_objectid_bool, map_saveid_objectid, map_class_saveid_count, extend_factor= 0.0):
    global low_pointed
    global total_objects
    global total_saved
    global GROUND_REMOVED_PLANE
    # Get objects
    objects = get_objects_from_label(label_file)

    # label_indices = list()
    # labels = np.zeros(pts_lidar.shape[0])
    # valid_pointcloud = ground_segmentation(pcd)
    
    # Get labels
    saveid = 0
    map_objectid_bool[log_idx] = {}
    map_saveid_objectid[log_idx] = {}
    map_class_saveid_count[log_idx] = {}
    map_objectid_bool[log_idx][pcd_idx] = {}
    map_saveid_objectid[log_idx][pcd_idx] = {}
    map_class_saveid_count[log_idx][pcd_idx] = {}
    low_pointed = 0
    
    if GROUND_REMOVED_PLANE:
        print("Before seg ", len(pts_lidar))
        segmented_pts_lidar, world_mask = ground_segmentation(pts_lidar)
        print("After seg ", len(segmented_pts_lidar))
        
        seg_folder = os.path.join(pcd_folder,"..", "seg_check")
        orig_folder = os.path.join(pcd_folder,"..", "orig_check")
        check_make_(seg_folder, False)
        check_make_(orig_folder, False)
        seg_pcd_save_path = os.path.join(seg_folder, "{0:06d}_{1:06d}.bin".format(log_idx, pcd_idx))
        orig_pcd_save_path = os.path.join(orig_folder, "{0:06d}_{1:06d}.bin".format(log_idx, pcd_idx))
        dump_pcd(seg_pcd_save_path, segmented_pts_lidar[:,:3])
        dump_pcd(orig_pcd_save_path, pts_lidar[:,:3])
        
        pts_lidar = segmented_pts_lidar
    
        map_seg2world = {}
        for each in np.arange(len(world_mask)):
            map_seg2world[each] = world_mask[each]

    for i, each in enumerate(objects):
        if each.cls_type not in object_dict.keys():
            map_objectid_bool[log_idx][pcd_idx][i] = 0
            continue
            
        else:
            object_id = object_dict[each.cls_type] #  type_to_id = {'VEHICLE': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
            saveid += 1
            
            each.pos = np.dot(each.pos, argo_to_kitti)

            # 3rd modification
            each.l = each.l + extend_factor * each.l
            each.w = each.w + extend_factor * each.w
            each.h = each.h + extend_factor * each.h

            indices_pruned, transformed_pointcloud_pruned = filter_pointcloud(each, np.copy(pts_lidar))
            indices_pruned = np.array(indices_pruned, dtype=int)
            if GROUND_REMOVED_PLANE:
                indices_pruned = np.array([map_seg2world[each] for each in indices_pruned])
            # labels[indices_pruned] = object_id
            
            pcd_save_path = os.path.join(pcd_folder, "{0:06d}_{1:06d}_{2}_{3}.bin".format(log_idx, idx, each.cls_type, saveid))
            ind_save_path = os.path.join(ind_folder, "{0:06d}_{1:06d}_{2}_{3}.txt".format(log_idx, idx, each.cls_type, saveid))
            if len(transformed_pointcloud_pruned)<100:
                low_pointed+=1
                continue
            dump_pcd(pcd_save_path, transformed_pointcloud_pruned)
            dump_txt(ind_save_path, indices_pruned)
            
            map_saveid_objectid[log_idx][pcd_idx][saveid] = 1
            if saveid in map_class_saveid_count[log_idx][pcd_idx].keys():
                map_class_saveid_count[log_idx][pcd_idx][saveid] +=1
            else:
                map_class_saveid_count[log_idx][pcd_idx][saveid] = 1
            map_objectid_bool[log_idx][pcd_idx][i] = 1
    # label_file.split("/")[-1]
    string = "Low points: {}/{} removed, {} saved {}".format(low_pointed, i, (i-low_pointed), ":" )
    total_objects += i
    total_saved +=i-low_pointed
    return map_saveid_objectid, map_class_saveid_count, map_objectid_bool, string
    
def cpy(source, destination):
    shutil.copy(source, destination) 
       
    
    
if __name__=="__main__":
    '''
    Data-format:
    argoverse-objects/
        train/
            pcd/
                000000_000000_car_1.bin
            indices/
                000000_000000_car_1.txt
            maps
                    ...
    '''
    base_root = os.getcwd()
    root_dir = os.path.join(base_root, "argoverse-tracking") + "/"
    save_h5_root = os.path.join(base_root, "argoverse-objects") + "/"
    
    dataset = ['train', 'val', 'test']
    folders = [os.path.join(root_dir, folder) for folder in ['train', 'val', 'test']]
    
    choice = 0 #args.dataset_choice
    folder_choice = folders[choice]
    dataset_choice = dataset[choice]
    
    root_dir_choice = os.path.join(root_dir, dataset_choice)
    save_dir_choice = os.path.join(save_h5_root, dataset_choice)
    save_dir_pcd = os.path.join(save_h5_root, dataset_choice, 'pcd')
    save_dir_ind = os.path.join(save_h5_root, dataset_choice, 'indices')
    check_make_(save_dir_pcd)
    check_make_(save_dir_ind)
    
    SEPERATE_ACTUAL = 0
    SEPERATE_CRUDE = 1
    GROUND_REMOVED = 1
    
    split = folder_choice[len(os.path.dirname(folder_choice))+1:]
    is_test = (split == 'test')
    
    am = ArgoverseMap()

    print("____________SPLIT IS : {} ______________".format(split))
    if split == 'train' or split == 'val':
        
        imageset_dir = os.path.join(root_dir,split)
        splitname = lambda x: [x[len(imageset_dir+"/"):-4].split("/")[0], x[len(imageset_dir+"/"):-4].split("/")[2].split("_")[1]]
        
        data_loader = ArgoverseTrackingLoader(os.path.join(root_dir,split))
        log_list = data_loader.log_list
        path_count = 0
        
        actual_idx_list = []
        logidx_to_count_map= {}
        log_to_count_map= {}
        map_objectid_bool = {}
        map_saveid_objectid = {}
        map_class_saveid_count = {}
        for log_id, log in enumerate(log_list):
            print("converting log {} {}".format(log, log_id))
            
            argoverse_data = data_loader.get(log)
            city_name = argoverse_data.city_name
            
            lidar_lst = data_loader.get(log).lidar_list
            label_lst = data_loader.get(log).label_list
            assert len(lidar_lst) == len(label_lst)
            pbar = tqdm(lidar_lst)
            for idx, each_path in enumerate(pbar):
                
                ide = log_id + idx
                # lidar_pts = argoverse_data.get_lidar(ide)
                lidar_pts = argoverse_data.get_lidar_ring(ide)
                
                if(GROUND_REMOVED):
                    city_to_egovehicle_se3 = argoverse_data.get_pose(ide)
                    roi_area_pts = city_to_egovehicle_se3.transform_point_cloud(lidar_pts[:, :3]) # more to city CS
                    roi_area_pts = am.remove_non_roi_points(roi_area_pts, city_name) # remove outside roi points
                    roi_area_pts = am.remove_ground_surface(roi_area_pts, city_name) # remove ground  
                    roi_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(
                        roi_area_pts
                    )# Back to lidar cs

                    x = np.array(roi_area_pts[:,0])[:, np.newaxis]
                    y = np.array(roi_area_pts[:,1])[:, np.newaxis]
                    z = np.array(roi_area_pts[:,2])[:, np.newaxis]
                    #i = np.array(roi_area_pts[:,3])[:, np.newaxis]
                    lidar_pts_seg = np.concatenate([x,y,z], axis = 1) 
                    
                    seg_folder = os.path.join(save_dir_pcd,"..", "seg_check")
                    orig_folder = os.path.join(save_dir_pcd,"..", "orig_check")
                    check_make_(seg_folder, False)
                    check_make_(orig_folder, False)
                    seg_pcd_save_path = os.path.join(seg_folder, "{0:06d}_{1:06d}.bin".format(log_id, idx))
                    orig_pcd_save_path = os.path.join(orig_folder, "{0:06d}_{1:06d}.bin".format(log_id, idx))
                    dump_pcd(seg_pcd_save_path, lidar_pts_seg[:,:3])
                    dump_pcd(orig_pcd_save_path, lidar_pts[:,:3])
                    
                    lidar_pts = lidar_pts_seg
                
                if SEPERATE_ACTUAL:
                    lidar_pts_up, lidar_pts_down = separate_pc(lidar_pts, tf_up_lidar, tf_down_lidar)
                if SEPERATE_CRUDE:
                    number_of_top_points = int(len(lidar_pts)/2)+3900
                    lidar_pts_up = lidar_pts[: number_of_top_points, :]
                    lidar_pts_down = lidar_pts[number_of_top_points: , :]

                print("Inital number of actual points: ", len(lidar_pts), " Top points :", len(lidar_pts_up))
                lidar_pts = lidar_pts_up
                map_objectid_bool, map_saveid_objectid, map_class_saveid_count, string = process_pcd(
                    lidar_pts, label_lst[idx], save_dir_pcd, save_dir_ind, log_id, idx, 
                    map_objectid_bool, map_saveid_objectid, map_class_saveid_count)
                pbar.set_description(string)
                
            actual_idx_list.extend([splitname(each) for each in lidar_lst])
            idx_list = np.arange(path_count, path_count + len(lidar_lst))
            logidx_to_count_map[log_id] = idx_list
            log_to_count_map[log] = idx_list
            path_count+=len(lidar_lst)
            if log_id == 0:
                break
        print("Total saved:{}, Total objects:{}, Ratio:{} ".format(total_saved, total_objects, total_saved*100/total_objects))
        dump_pickle(os.path.join(save_dir_choice, "logidx_to_count_map"), logidx_to_count_map)
        dump_pickle(os.path.join(save_dir_choice, "log_to_count_map"), log_to_count_map)
        dump_pickle(os.path.join(save_dir_choice, "map_objectid_bool"), map_objectid_bool)
        dump_pickle(os.path.join(save_dir_choice, "map_saveid_objectid"), map_saveid_objectid)
        dump_pickle(os.path.join(save_dir_choice, "map_class_saveid_count"), map_class_saveid_count)
        dump_pickle(os.path.join(save_dir_choice, "actual_idx_list"), actual_idx_list)


        
    
