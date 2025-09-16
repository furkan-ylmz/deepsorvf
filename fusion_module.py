"""
from fusion import Fusion

# Fusion modülü oluştur
fusion = Fusion(
    max_distance=200,        # Maksimum eşleşme mesafesi (piksel)
    image_shape=[1920, 1080], # Video çözünürlüğü [width, height]
    time_interval=40         # Zaman aralığı (ms)
)

# AIS ve VIS verilerini fusion et
matched_data, fusion_results = fusion.fusion(
    AIS_vis=ais_data,        # AIS trajectory DataFrame
    AIS_cur=ais_current,     # Mevcut AIS verileri
    Vis_tra=vis_data,        # VIS trajectory DataFrame  
    Vis_cur=vis_current,     # Mevcut VIS verileri
    timestamp=current_timestamp   # Mevcut zaman damgası
)
"""
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment as linear_assignment
from fastdtw import fastdtw

class Fusion:
    def __init__(self, max_distance=200, image_shape=[1920, 1080], time_interval=40):
        """
        Args:
            max_distance (int): max_dis parametresi
            image_shape (list): im_shape parametresi
            time_interval (int): t parametresi
        """
        self.max_dis = max_distance
        self.im_shape = image_shape
        self.t = time_interval
        
        self.bin_num = 1
        self.fog_num = 2
        
        self.mat_cur = pd.DataFrame(columns=['ID/mmsi','timestamp', 'match'])
        self.mat_list = pd.DataFrame(columns=['ID', 'mmsi', 'lon', 'lat', 'speed', 'course', 'heading', 'type', 'timestamp'])
        self.bin_cur = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
    
    def _reduce_by_half(self, x):
        return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

    def _angle(self, v1, v2):
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

    def _DTW_fast(self, traj0, traj1):
        if len(traj0)>1 and len(traj1)>1:
            theta = self._angle(traj0, traj1)
            traj0 = self._reduce_by_half(traj0)
            traj1 = self._reduce_by_half(traj1)
        else:
            theta = 0
        
        d, path = fastdtw(traj0, traj1, dist=euclidean)
        
        return d*math.exp(theta)

    def _traj_group(self, df_data, df_dataCur, kind):
        """
        return: trajData_list, trajLabel_list, trajInf_list
        """
        
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
    
    def initialization(self, AIS_list, VIS_list):
        mat_las = self.mat_cur
        bin_las = mat_las[mat_las['match'] > self.bin_num]
        mat_cur = pd.DataFrame(columns=['ID/mmsi','timestamp', 'match'])
        bin_cur = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        mat_list = pd.DataFrame(columns=['ID', 'mmsi',
                                'lon', 'lat', 'speed', 'course', 'heading', 'type', 'x1',
                                    'y1', 'w', 'h', 'timestamp'])
        
        return mat_cur, bin_cur, mat_las, bin_las, mat_list
    
    def cal_similarity(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las):
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
                    theta = self._angle(VIS_list[i], AIS_list[j])
                    
                    x_VIS = VIS_list[i][-1][0]
                    y_VIS = VIS_list[i][-1][1]
                    x_AIS = AIS_list[j][-1][0]
                    y_AIS = AIS_list[j][-1][1]
                    dis = ((x_VIS-x_AIS)**2+(y_VIS-y_AIS)**2)**0.5
                    
                    if dis < self.max_dis and theta < math.pi*(7/8):  # Daha geniş açı toleransı
                        matrix_S[i][j] = self._DTW_fast(VIS_list[i], AIS_list[j])
                    else:
                        matrix_S[i][j] = 1000000000
                
                elif cur_IDmmsi in binIDmmsi:
                    matrix_S[i][j] = 0-int(bin_las[bin_las['ID/mmsi'] == cur_IDmmsi]['match'].values)*100
                
                else:
                    matrix_S[i][j] = 1000000000
        return matrix_S
    
    def data_filter(self, row_ind, col_ind, VIS_list, AIS_list):
        matches = []

        for row, col in zip(row_ind, col_ind):
            theta = self._angle(VIS_list[row], AIS_list[col])
            
            x_VIS = VIS_list[row][-1][0]
            y_VIS = VIS_list[row][-1][1]
            x_AIS = AIS_list[col][-1][0]
            y_AIS = AIS_list[col][-1][1]
            dis = ((x_VIS-x_AIS)**2+(y_VIS-y_AIS)**2)**0.5
            
            if dis < self.max_dis and theta < math.pi*(5/6): # 
                matches.append((row, col))
        return matches
    
    def save_data(self, mat_cur, bin_cur, mat_las, bin_las, mat_list,
                  matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp):
        
        for i in range(len(matches)):
            v_loc, a_loc = matches[i][0], matches[i][1]
            ID = int(VIS_IDlist[v_loc])
            MMSI = int(AIS_MMSIlist[a_loc])
            ID_MMSI = str(ID)+'/'+str(MMSI)

            lon = AInf_list[a_loc][-1][1]
            lat = AInf_list[a_loc][-1][2]
            speed = AInf_list[a_loc][-1][3]
            course = AInf_list[a_loc][-1][4]
            heading = AInf_list[a_loc][-1][5]
            types = AInf_list[a_loc][-1][6]
            time = AInf_list[a_loc][-1][9]
            
            x1 = max(VInf_list[v_loc][-1][1],0)
            y1 = max(VInf_list[v_loc][-1][2],0)
            x2 = min(VInf_list[v_loc][-1][3],self.im_shape[0])
            y2 = min(VInf_list[v_loc][-1][4],self.im_shape[1])
            w = abs(x2-x1)
            h = abs(y2-y1)
            
            new_row = pd.DataFrame([{'ID':ID,'mmsi':MMSI,'lon':lon,'lat':lat,
                'speed':speed,'course': course,'heading':heading,'type':types,'x1':x1,'y1':y1,
                    'w':w,'h':h,'timestamp':time}])
            mat_list = pd.concat([mat_list, new_row], ignore_index=True)
            
            if ID_MMSI in mat_las['ID/mmsi'].values:
                match = mat_las[mat_las['ID/mmsi'] == ID_MMSI]['match'].values[0]+1
                new_row = pd.DataFrame([{'ID/mmsi':str(ID)+'/'+str(MMSI),
                                           'timestamp':time,'match':match}])
                mat_cur = pd.concat([mat_cur, new_row], ignore_index=True)
            
            else:  
                new_row = pd.DataFrame([{'ID/mmsi':str(ID)+'/'+str(MMSI),
                                           'timestamp':time,'match':1}])
                mat_cur = pd.concat([mat_cur, new_row], ignore_index=True)
        
        for ind, inf in bin_las.iterrows():
            ID_MMSI = inf['ID/mmsi']
            ID, MMSI = [int(x) for x in ID_MMSI.split('/')]
            time = inf['timestamp']
            if MMSI in AIS_MMSIlist and ID_MMSI not in mat_cur['ID/mmsi'].values and timestamp//1000-time < self.fog_num:
                mat_cur = pd.concat([mat_cur, pd.DataFrame([inf])], ignore_index=True)
        
        for ind, inf in mat_cur.iterrows():
            ID, MMSI = [int(x) for x in inf['ID/mmsi'].split('/')]
            if inf['match'] > self.bin_num:
                new_row = pd.DataFrame([{'ID': ID, 'mmsi': MMSI,
                  'timestamp': int(inf['timestamp']), 'match': int(inf['match'])}])
                bin_cur = pd.concat([bin_cur, new_row], ignore_index=True)

        return mat_list, mat_cur, bin_cur
    
    def traj_match(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp):
        
        mat_cur, bin_cur, mat_las, bin_las, mat_list = self.initialization(AIS_list, VIS_list)
        
        matrix_S = self.cal_similarity(AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las)
        
        row_ind, col_ind = linear_assignment(matrix_S)
        
        matches = self.data_filter(row_ind, col_ind, VIS_list, AIS_list)

        # matric = pd.DataFrame(matrix_S,columns=AIS_MMSIlist,index=VIS_IDlist)
        
        mat_list, mat_cur, bin_cur = self.save_data(mat_cur, bin_cur, mat_las, bin_las,
                                    mat_list, matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp)

        return mat_list, mat_cur, bin_cur
    
    def fusion(self, AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp):
        if timestamp % 1000 < self.t:
            
            AIS_list, AIS_MMSIlist, AInf_list = self._traj_group(AIS_vis, AIS_cur, 'AIS')
            VIS_list, VIS_IDlist, VInf_list = self._traj_group(Vis_tra, Vis_cur, 'VIS')

            self.mat_list, self.mat_cur, self.bin_cur = self.traj_match(AIS_list,
                                AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp)

        return self.mat_list, self.bin_cur
    
def example_usage():
    print("🚀 Standalone Fusion modülü test örneği")
    print("   (Orijinal FUSPRO algoritması)")
    
    # Fusion modülü oluştur
    fusion = Fusion(
        max_distance=200,
        image_shape=[1920, 1080],
        time_interval=40
    )
    
    # Örnek AIS verileri (FUSPRO formatında)
    ais_trajectories = pd.DataFrame({
        'mmsi': [123456, 123456, 789012, 789012],      # sütun 0
        'lon': [114.327, 114.328, 114.330, 114.331],   # sütun 1
        'lat': [30.600, 30.601, 30.602, 30.603],       # sütun 2
        'speed': [5.2, 5.1, 8.7, 8.9],                 # sütun 3
        'course': [45.0, 46.0, 120.0, 121.0],          # sütun 4
        'heading': [44.0, 45.0, 119.0, 120.0],         # sütun 5
        'type': [6, 6, 7, 7],                          # sütun 6
        'x': [100, 105, 200, 205],                     # sütun 7 (pixel X)
        'y': [100, 102, 150, 152],                     # sütun 8 (pixel Y)
        'timestamp': [1000, 1040, 1000, 1040],         # sütun 9
    })
    
    ais_current = pd.DataFrame({
        'mmsi': [123456, 789012],
        'timestamp': [1040, 1040]
    })
    
    # Örnek VIS verileri (FUSPRO formatında)
    vis_trajectories = pd.DataFrame({
        'ID': [1, 1, 2, 2],                           # sütun 0
        'x1': [95, 100, 195, 200],                    # sütun 1
        'y1': [95, 97, 145, 147],                     # sütun 2
        'x2': [145, 150, 235, 240],                   # sütun 3
        'y2': [125, 127, 170, 172],                   # sütun 4
        'x': [120, 125, 215, 220],                    # sütun 5 (center X)
        'y': [110, 112, 157, 159],                    # sütun 6 (center Y)
        'timestamp': [1000, 1040, 1000, 1040],        # sütun 7+
    })
    
    vis_current = pd.DataFrame({
        'ID': [1, 2],
        'timestamp': [1040, 1040]
    })
    
    # Fusion işlemi - orijinal interface
    matched_data, fusion_results = fusion.fusion(
        AIS_vis=ais_trajectories,
        AIS_cur=ais_current,
        Vis_tra=vis_trajectories,
        Vis_cur=vis_current,
        timestamp=35  # 40ms'den küçük değer
    )
    
    print("\n📊 Fusion Sonuçları (Orijinal Format):")
    print("Mat_list (tüm eşleşmeler):")
    print(matched_data)
    print("\nBin_cur (aktif binding'ler):")
    print(fusion_results)
    
    # Alternatif interface test
    print("\n🔄 Fusion testi:")
    matched_data2, fusion_results2 = fusion.fusion(
        AIS_vis=ais_trajectories,
        AIS_cur=ais_current,
        Vis_tra=vis_trajectories,
        Vis_cur=vis_current,
        timestamp=75  # 40ms'den büyük değer
    )
    
    print("Process sonuçları:")
    print("Matched:", len(matched_data2), "Bindings:", len(fusion_results2))


if __name__ == "__main__":
    example_usage()
