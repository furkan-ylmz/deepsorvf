import pandas as pd
from geopy.distance import geodesic
import pyproj
from math import radians, cos, sin, tan, atan2, degrees
import math
import socket
from datetime import datetime
from pyais import decode

class AISPRO(object):
    def __init__(self, ais_host, ais_port, im_shape, t):
        self.ais_host = ais_host
        self.ais_port = ais_port
        self.im_shape = im_shape
        self.max_dis  = 2*1852
        self.t        = t
        self.time_lim = 2
        self.AIS_cur  = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','timestamp'])
        self.AIS_vis  = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','x','y','timestamp'])
    
    def initialization(self):
        
        AIS_las = self.AIS_cur
        AIS_vis = self.AIS_vis
        AIS_cur = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','timestamp'])
        return AIS_cur, AIS_las, AIS_vis
    
    def read_ais(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ais_host, self.ais_port))
        sock_file = sock.makefile('r')

        while True:
            line = sock_file.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("!AIVDM") or line.startswith("!AIVDO"):
                ais_row = self.parse_nmea(line)
                if ais_row is not None:
                    yield ais_row


    def parse_nmea(self, nmea_sentence):
        try:
            msg = decode(nmea_sentence)
            return {
                'mmsi': msg.mmsi,
                'lon': msg.lon,
                'lat': msg.lat,
                'speed': msg.speed,
                'course': msg.course,
                'heading': msg.heading,
                'type': getattr(msg, 'ship_type', None),
                'timestamp': int(datetime.utcnow().timestamp() * 1000)  # ms cinsinden
            }
        except Exception as e:
            print(f"NMEA parse hatası: {e}")
            return None

        
    def data_tran(self, AIS_cur, AIS_vis, camera_para, timestamp):

        AIS_vis, AIS_vis_cur = self.transform(AIS_cur, AIS_vis, camera_para, self.im_shape)

        # self.AIS_pre = self.AIS_pre.append(self.AIS_cur, ignore_index=True)
        AIS_vis = pd.concat([AIS_vis, AIS_vis_cur], ignore_index=True)
        
        # self.AIS_pre = self.AIS_pre.drop(self.AIS_pre[self.AIS_pre['time'] < (timestamp // 1000 - self.time_lim * 60)].index)
        AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['timestamp'] < (timestamp//1000 - self.time_lim * 60)].index)
        return AIS_vis
    
    def ais_pro(self, AIS_cur, AIS_las, AIS_vis, camera_para, timestamp):
        rows = []
        for ais_row in self.read_ais():  # read_ais sonsuz döngü ile sürekli veri üretir
            rows.append(ais_row)

            # Her yeni satır geldiğinde işleme al
            AIS_read = pd.DataFrame(rows)
            AIS_read = self.data_coarse_process(AIS_read, AIS_las, camera_para, self.max_dis)
            AIS_cur = self.data_pred(AIS_cur, AIS_read, AIS_las, timestamp)
            AIS_vis = self.data_tran(AIS_cur, AIS_vis, camera_para, timestamp)

            # İstersen burada rows'u temizleyebilirsin ki hafıza şişmesin
            rows.clear()

            # AIS_vis ve AIS_cur'u her adımda güncel tut
            self.AIS_vis, self.AIS_cur = AIS_vis, AIS_cur

    
    def process(self, camera_para, timestamp):
        #if timestamp % 1000 < self.t:
        AIS_cur, AIS_las, AIS_vis = self.initialization()
        self.AIS_vis, self.AIS_cur = self.ais_pro(AIS_cur, AIS_las, AIS_vis, camera_para, timestamp)
        return self.AIS_vis, self.AIS_cur


    @staticmethod
    def count_distance(point1, point2, Type='m'):

        distance = geodesic(point1, point2).m
        if Type == 'nm':
            distance = distance * 0.00054
        return distance

    @staticmethod
    def getDegree(latA, lonA, latB, lonB):

        radLatA = radians(latA)
        radLonA = radians(lonA)
        radLatB = radians(latB)
        radLonB = radians(lonB)
        dLon = radLonB - radLonA
        y = sin(dLon) * cos(radLatB)
        x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
        brng = degrees(atan2(y, x))
        brng = (brng + 360) % 360
        return brng

    @staticmethod
    def visual_transform(lon_v, lat_v, camera_para, shape):

        lon_cam = camera_para[0]
        lat_cam = camera_para[1]
        shoot_hdir = camera_para[2]
        shoot_vdir = camera_para[3]
        height_cam = camera_para[4]
        FOV_hor = camera_para[5]
        FOV_ver = camera_para[6]
        width_pic = shape[0]
        height_pic = shape[1]
        f_x = camera_para[7]
        f_y = camera_para[8]
        u0  = camera_para[9]
        v0  = camera_para[10]

        D_abs = AISPRO.count_distance((lat_cam, lon_cam), (lat_v, lon_v)) 
        relative_angle = AISPRO.getDegree(lat_cam, lon_cam, lat_v, lon_v)
        Angle_hor = relative_angle - shoot_hdir

        if Angle_hor < -180:
            Angle_hor = Angle_hor + 360
        elif Angle_hor > 180:
            Angle_hor = Angle_hor - 360

        hor_rad = radians(Angle_hor)
        shv_rad = radians(-shoot_vdir)
        Z_w = D_abs*cos(hor_rad)
        X_w = D_abs*sin(hor_rad)
        Y_w = height_cam
        Z = Z_w/cos(shv_rad)+(Y_w-Z_w*tan(shv_rad))*sin(shv_rad)
        X = X_w
        Y = (Y_w-Z_w*tan(shv_rad))*cos(shv_rad)

        target_x = int(f_x*X/Z+u0)
        target_y = int(f_y*Y/Z+v0)
        
        #Angle_ver = 90 + shoot_vdir - math.degrees(math.atan(D_abs / height_cam))
        #target_x1 = int(width_pic // 2 + width_pic * Angle_hor / FOV_hor)
        #target_y1 = int(height_pic // 2 + height_pic * Angle_ver / FOV_ver)

        return target_x, target_y

    def data_filter(self, ais, camera_para):

        lon_cam = camera_para[0]
        lat_cam = camera_para[1]
        shoot_hdir = camera_para[2]
        shoot_vdir = camera_para[3]
        height_cam = camera_para[4]
        FOV_hor = camera_para[5]
        FOV_ver = camera_para[6]

        lon, lat = ais['lon'], ais['lat']
        D_abs = self.count_distance((lat_cam,lon_cam),(lat,lon))
        angle = self.getDegree(lat_cam, lon_cam, lat, lon)
        in_angle = abs(shoot_hdir - angle) if abs(shoot_hdir - angle) < 180 else 360 - abs(shoot_hdir - angle)

        if 90 + shoot_vdir - FOV_ver / 2 < math.degrees(math.atan(D_abs / height_cam)):
            
            if in_angle <= (FOV_hor / 2 + 8):  
                return 'transform'
            
            elif in_angle > (FOV_hor / 2 + 8):
                return 'visTraj_del'
            
            if in_angle > (FOV_hor / 2 + 12):
                return 'ais_del'

    def transform(self, AIS_current, AIS_vis, camera_para, shape):

        AIS_visCurrent = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','x','y','timestamp'])
        
        for index, ais in AIS_current.iterrows():

            flag = self.data_filter(ais, camera_para)
            if flag == 'transform':
                x, y = self.visual_transform(ais['lon'], ais['lat'], camera_para, shape)
                ais['x'], ais['y'] = x, y
                AIS_visCurrent = pd.concat([AIS_visCurrent, ais.to_frame().T], ignore_index=True)
            
            elif flag == 'visTraj_del' or flag == 'ais_del':
                AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['mmsi'] == ais['mmsi']].index)
        return AIS_vis, AIS_visCurrent

    def data_pre(self, ais, timestamp):

        if ais['speed'] == 0:
            ais['timestamp'] = timestamp
        
        else:
            geo_d = pyproj.Geod(ellps="WGS84")
            
            distance = ais['speed'] * ((timestamp - ais['timestamp']) / 3600) * 1852
            ais['timestamp'] = timestamp

            ais['lon'], ais['lat'], c = geo_d.fwd(ais['lon'], ais['lat'], ais['course'], distance)
        return ais

    def data_pred(self, AIS_cur, AIS_read, AIS_las, timestamp):

        # Time offset correction - AIS data is ~5 hours behind video
        TIME_OFFSET = 5 * 3600 * 1000  # 5 hours in milliseconds

        for index, ais in AIS_read.iterrows():
            # Apply time offset to AIS timestamp
            ais['timestamp'] = ais['timestamp'] + TIME_OFFSET
            ais['timestamp'] = round(ais['timestamp']/1000)
            
            if ais['timestamp'] == int(timestamp//1000):
                AIS_cur = pd.concat([AIS_cur, ais.to_frame().T], ignore_index=True)
            
            else:
                predicted_data = self.data_pre(ais, timestamp//1000)
                AIS_cur = pd.concat([AIS_cur, predicted_data.to_frame().T], ignore_index=True)
        
        for index, ais in AIS_las.iterrows():
            if ais['mmsi'] not in AIS_cur['mmsi'].values:
                predicted_data = self.data_pre(ais, timestamp//1000)
                AIS_cur = pd.concat([AIS_cur, predicted_data.to_frame().T], ignore_index=True)
        return AIS_cur

    def data_coarse_process(self, AIS_current,AIS_last,camera_para,max_dis):

        camera_loc = (camera_para[1], camera_para[0])
        
        for index, ais in AIS_current.iterrows():
            flag = 0
            
            if ais['mmsi'] / 100000000 < 1 or ais['mmsi'] / 100000000 >= 10 or\
                ais['lon'] == -1 or ais['lat'] == -1 or ais['speed'] == -1 or\
                    ais['course'] == -1 or ais['course'] == 360 or ais['heading'] == -1 or ais['lon'] > 180 or\
                        ais['lon'] < 0 or ais['lat'] > 90 or ais['lat'] < 0 or ais['speed'] <= 0.3:
                AIS_current = AIS_current.drop(index=index)
                continue

            if ais['mmsi'] in AIS_last['mmsi'].values:
                temp = AIS_last[AIS_last.mmsi == ais['mmsi']]
                if abs(ais['lon'] - temp['lon'].values[-1]) >= 1 \
                    or abs(ais['lat'] - temp['lat'].values[-1]) >= 1 \
                        or abs(ais['speed'] - temp['speed'].values[-1]) >= 7:
                    AIS_current = AIS_current.drop(index=index)
                    continue
    
            ship_loc = (ais['lat'], ais['lon'])
            dis = self.count_distance(camera_loc, ship_loc, Type='m')
            if dis > max_dis or self.data_filter(ais, camera_para) == 'ais_del':
                AIS_current = AIS_current.drop(index=index)
        return AIS_current


# Example usage - original interface
if __name__ == "__main__":
    print("🚀 Original AISPRO test")
    
    # Original constructor interface
    aispro = AISPRO(
        ais_host="127.0.0.1",
        ais_port=10110,
        im_shape=[1920, 1080],
        t=50
    )
    
    # Original camera parameters format
    camera_para = [114.32722222222222, 30.60027777777778, 352, -4, 20, 55, 30.94, 2391.26, 2446.89, 1305.04, 855.214]
    timestamp = 1654336512000
    Time_name = "2022_06_04_12_05_12.csv"
    
    try:
        # Original method call
        AIS_vis, AIS_cur = aispro.process(camera_para, timestamp)
        print(f"✅ AIS processed: {len(AIS_vis)} visible, {len(AIS_cur)} current")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("(This is expected if AIS files or dependencies are missing)")