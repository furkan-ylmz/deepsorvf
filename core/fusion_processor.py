from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
import math
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

def angle(v1, v2):
    
    if len(v1) >= 10:
        dx1 = v1[-1][0] - v1[-10][0]
        dy1 = v1[-1][1] - v1[-10][1]
    elif len(v1) < 10:
        dx1 = v1[-1][0] - v1[0][0]
        dy1 = v1[-1][1] - v1[0][1]
    if len(v2) >= 5:
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]
    elif len(v2) < 5:
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]
    
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > math.pi:
            included_angle = math.pi*2 - included_angle
    return included_angle

from core.ekf_fusion import EKFFusionManager

def multi_feature_dtw(vis_traj, ais_traj, vis_inf=None, ais_inf=None, w_p=0.50, w_v=0.25, w_theta=0.15, w_s=0.10):
    """
    Multi-Feature Dynamic Time Warping (MF-DTW) calculating comprehensive trajectory similarity:
    1. Spatial Position Distance (D_pos)
    2. Velocity Vector Difference (D_vel)
    3. Heading / Course Angular Deviation (D_theta)
    4. Bounding Box Aspect Ratio Consistency (D_scale)
    """
    if len(vis_traj) == 0 or len(ais_traj) == 0:
        return 1e9
        
    theta = angle(vis_traj, ais_traj)
    
    # 1. Spatial DTW
    v_pos = __reduce_by_half(vis_traj) if len(vis_traj) > 1 else vis_traj
    a_pos = __reduce_by_half(ais_traj) if len(ais_traj) > 1 else ais_traj
    d_pos, _ = fastdtw(v_pos, a_pos, dist=euclidean)
    
    # 2. Velocity Difference
    d_vel = 0.0
    if len(vis_traj) >= 2 and len(ais_traj) >= 2:
        v_vel = np.diff(vis_traj, axis=0)
        a_vel = np.diff(ais_traj, axis=0)
        d_vel, _ = fastdtw(v_vel, a_vel, dist=euclidean)
    
    # 3. Angular Deviation (0 to 2)
    d_theta = 1.0 - math.cos(theta)
    
    # 4. Aspect Ratio Consistency
    d_scale = 0.0
    if vis_inf is not None and len(vis_inf) > 0 and ais_inf is not None and len(ais_inf) > 0:
        try:
            w_box = abs(vis_inf[-1][3] - vis_inf[-1][1])
            h_box = abs(vis_inf[-1][4] - vis_inf[-1][2])
            aspect_vis = w_box / max(h_box, 1.0)
            d_scale = abs(aspect_vis - 1.8) / 1.8
        except Exception:
            d_scale = 0.0
            
    total_cost = (w_p * d_pos + w_v * d_vel * 10.0 + w_theta * (d_theta * 100.0) + w_s * (d_scale * 50.0))
    return total_cost

def DTW_fast(traj0, traj1):
    if len(traj0)>1 and len(traj1)>1:
        theta = angle(traj0, traj1)
        traj0 = __reduce_by_half(traj0)
        traj1 = __reduce_by_half(traj1)
    else:
        theta = 0
    
    d, path = fastdtw(traj0, traj1, dist=euclidean)
    return d*math.exp(theta)

def traj_group(df_data, df_dataCur,  kind):
    trajData_list = []  
    trajLabel_list = []  
    trajInf_list = []  
    
    if kind == 'AIS':
        grouped = df_data.groupby('mmsi')
        for value, group in grouped:
            if value in df_dataCur['mmsi'].tolist():
                traj = group.values
                trajData_list.append(np.array(traj[:, 7:9]))
                trajLabel_list.append(int(traj[0, 0]))
                trajInf_list.append(traj)
    
    elif kind == 'VIS':
        grouped = df_data.groupby('ID')
        for value, group in grouped:
            if value in df_dataCur['ID'].tolist():
                traj = group.values
                trajData_list.append(np.array(traj[:, 5:7]))
                trajLabel_list.append(int(traj[0][0]))
                trajInf_list.append(traj)

    return trajData_list, trajLabel_list, trajInf_list

class FUSPRO(object):
    def __init__(self, max_dis, im_shape, t, w_p=0.50, w_v=0.25, w_theta=0.15, w_s=0.10):
        self.max_dis = max_dis
        self.im_shape = im_shape
        self.bin_num = 1
        self.fog_num = 2
        self.t = t
        self.w_p = w_p
        self.w_v = w_v
        self.w_theta = w_theta
        self.w_s = w_s
        self.ekf_manager = EKFFusionManager()
        self.mat_cur  = pd.DataFrame(pd.DataFrame(columns=['ID/mmsi','timestamp', 'match']))
        self.mat_list = pd.DataFrame(columns=['ID', 'mmsi', 'lon', 'lat', 'speed', 'course', 'heading', 'type', 'timestamp'])
        self.bin_cur  = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    def initialization(self, AIS_list, VIS_list):
        mat_las   = self.mat_cur
        bin_las   = mat_las[mat_las['match'] > self.bin_num]
        mat_cur   = pd.DataFrame(pd.DataFrame(columns=['ID/mmsi','timestamp', 'match']))
        bin_cur   = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        mat_list  = pd.DataFrame(columns=['ID', 'mmsi', 'lon', 'lat', 'speed', 'course', 'heading', 'type', 'x1', 'y1', 'w', 'h', 'timestamp'])
        return mat_cur, bin_cur, mat_las, bin_las, mat_list
    
    def cal_similarity(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las, AInf_list=None, VInf_list=None):
        matrix_S = np.zeros((len(VIS_list), len(AIS_list)))
        binIDmmsi, bin_MMSI, bin_ID = [], [], []

        if len(bin_las)!=0:
            grouped = bin_las.groupby('ID/mmsi')
            for value, group in grouped:
                ID, MMSI = value.split('/')
                bin_ID.append(int(ID))
                bin_MMSI.append(int(MMSI))
                binIDmmsi.append(value)
                
        for i in range(len(VIS_list)):
            for j in range(len(AIS_list)):
                cur_ID, cur_mmsi = VIS_IDlist[i], AIS_MMSIlist[j]
                cur_IDmmsi = str(int(cur_ID))+'/'+str(int(cur_mmsi))
                
                if int(cur_mmsi) not in bin_MMSI and int(cur_ID) not in bin_ID:
                    theta = angle(VIS_list[i], AIS_list[j])
                    
                    x_VIS = VIS_list[i][-1][0]
                    y_VIS = VIS_list[i][-1][1]
                    x_AIS = AIS_list[j][-1][0]
                    y_AIS = AIS_list[j][-1][1]
                    dis   = ((x_VIS-x_AIS)**2+(y_VIS-y_AIS)**2)**0.5
                    
                    if dis < self.max_dis and theta < math.pi*(7/8):
                        v_inf = VInf_list[i] if VInf_list is not None else None
                        a_inf = AInf_list[j] if AInf_list is not None else None
                        matrix_S[i][j] = multi_feature_dtw(VIS_list[i], AIS_list[j], v_inf, a_inf, self.w_p, self.w_v, self.w_theta, self.w_s)
                    else:
                        matrix_S[i][j] = 1000000000
                
                elif cur_IDmmsi in binIDmmsi:
                    matched_arr = bin_las[bin_las['ID/mmsi'] == cur_IDmmsi]['match'].values
                    match_val = int(matched_arr[0]) if len(matched_arr) > 0 else 1
                    # Temporal consistency bonus
                    matrix_S[i][j] = 0 - match_val * 150
                
                else:
                    matrix_S[i][j] = 1000000000
        return matrix_S
    
    def data_filter(self, row_ind, col_ind, VIS_list, AIS_list):
        
        matches = []

        for row, col in zip(row_ind, col_ind):
            
            theta = angle(VIS_list[row], AIS_list[col])
            
            x_VIS = VIS_list[row][-1][0]
            y_VIS = VIS_list[row][-1][1]
            x_AIS = AIS_list[col][-1][0]
            y_AIS = AIS_list[col][-1][1]
            dis   = ((x_VIS-x_AIS)**2+(y_VIS-y_AIS)**2)**0.5
            
            if dis < self.max_dis and theta < math.pi*(5/6): # 
                matches.append((row, col))
        return matches
    
    def save_data(self, mat_cur, bin_cur, mat_las, bin_las, mat_list,\
                  matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp):
        
        for i in range(len(matches)):
            v_loc, a_loc = matches[i][0],matches[i][1]
            ID           = int(VIS_IDlist[v_loc])
            MMSI         = int(AIS_MMSIlist[a_loc])
            ID_MMSI      = str(ID)+'/'+str(MMSI)

            lon          = AInf_list[a_loc][-1][1]
            lat          = AInf_list[a_loc][-1][2]
            speed        = AInf_list[a_loc][-1][3]
            course       = AInf_list[a_loc][-1][4]
            heading      = AInf_list[a_loc][-1][5]
            types        = AInf_list[a_loc][-1][6]
            time         = AInf_list[a_loc][-1][9]
            
            x1           = max(VInf_list[v_loc][-1][1],0)
            y1           = max(VInf_list[v_loc][-1][2],0)
            x2           = min(VInf_list[v_loc][-1][3],self.im_shape[0])
            y2           = min(VInf_list[v_loc][-1][4],self.im_shape[1])
            w            = abs(x2-x1)
            h            = abs(y2-y1)
            
            new_row = pd.DataFrame([{'ID':ID,'mmsi':MMSI,'lon':lon,'lat':lat,\
                'speed':speed,'course': course,'heading':heading,'type':types,'x1':x1,'y1':y1,\
                    'w':w,'h':h,'timestamp':time}])
            mat_list = pd.concat([mat_list, new_row], ignore_index=True)
            
            if ID_MMSI in mat_las['ID/mmsi'].values:
                match = mat_las[mat_las['ID/mmsi'] == ID_MMSI]['match'].values[0]+1
                new_row = pd.DataFrame([{'ID/mmsi':str(ID)+'/'+str(MMSI),\
                                           'timestamp':time,'match':match}])
                mat_cur = pd.concat([mat_cur, new_row], ignore_index=True)
            
            else:  
                new_row = pd.DataFrame([{'ID/mmsi':str(ID)+'/'+str(MMSI),\
                                           'timestamp':time,'match':1}])
                mat_cur = pd.concat([mat_cur, new_row], ignore_index=True)
        
        for ind, inf in bin_las.iterrows():
            ID_MMSI = inf['ID/mmsi']
            ID, MMSI = [int(x) for x in ID_MMSI.split('/')]
            time    = inf['timestamp']
            if MMSI in AIS_MMSIlist and ID_MMSI not in mat_cur['ID/mmsi'].values\
                                                and timestamp//1000-time < self.fog_num:
                mat_cur = pd.concat([mat_cur, pd.DataFrame([inf])], ignore_index=True)
        
        for ind, inf in mat_cur.iterrows():
            ID, MMSI = [int(x) for x in inf['ID/mmsi'].split('/')]
            if inf['match'] > self.bin_num:
                new_row = pd.DataFrame([{'ID': ID, 'mmsi': MMSI,\
                  'timestamp': int(inf['timestamp']), 'match': int(inf['match'])}])
                bin_cur = pd.concat([bin_cur, new_row], ignore_index=True)

        return mat_list, mat_cur, bin_cur
    
    def traj_match(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp):
        mat_cur, bin_cur, mat_las, bin_las, mat_list = self.initialization(AIS_list, VIS_list)
        
        matrix_S = self.cal_similarity(AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las, AInf_list, VInf_list)
        
        row_ind, col_ind = linear_assignment(matrix_S)
        
        matches = self.data_filter(row_ind, col_ind, VIS_list, AIS_list)

        mat_list, mat_cur, bin_cur = self.save_data(mat_cur, bin_cur, mat_las, bin_las, mat_list, matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp)

        return mat_list, mat_cur, bin_cur
    
    def fusion(self, AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp):
        AIS_list, AIS_MMSIlist, AInf_list = traj_group(AIS_vis, AIS_cur, 'AIS')
        VIS_list, VIS_IDlist, VInf_list = traj_group(Vis_tra, Vis_cur, 'VIS')

        self.mat_list, self.mat_cur, self.bin_cur = self.traj_match(AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp)
        
        # Update EKF State Estimation with latest matched records
        if len(self.mat_list) > 0:
            self.ekf_manager.update_fusion(self.mat_list, AIS_cur, Vis_cur)
        else:
            self.ekf_manager.predict_all()
            
        return self.mat_list, self.bin_cur

    def get_ekf_states(self):
        """Return list of smoothed EKF vessel states."""
        return self.ekf_manager.get_smoothed_tracks()

