import pandas as pd
from geopy.distance import geodesic
import pyproj
from math import radians, cos, sin, asin, sqrt, tan, atan2, degrees
import math
import numpy as np
import cv2
from IPython import embed
import os

def count_distance(point1, point2, Type='m'):

    distance = geodesic(point1, point2).m
    if Type == 'nm':
        distance = distance * 0.00054
    return distance

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

    D_abs = count_distance((lat_cam, lon_cam), (lat_v, lon_v)) 
    relative_angle = getDegree(lat_cam, lon_cam, lat_v, lon_v)
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

def data_filter(ais, camera_para):

    lon_cam = camera_para[0]
    lat_cam = camera_para[1]
    shoot_hdir = camera_para[2]
    shoot_vdir = camera_para[3]
    height_cam = camera_para[4]
    FOV_hor = camera_para[5]
    FOV_ver = camera_para[6]

    lon, lat = ais['lon'], ais['lat']
    D_abs = count_distance((lat_cam,lon_cam),(lat,lon))
    angle = getDegree(lat_cam, lon_cam, lat, lon)
    in_angle = abs(shoot_hdir - angle) if abs(shoot_hdir - angle) < 180 else 360 - abs(shoot_hdir - angle)

    if 90 + shoot_vdir - FOV_ver / 2 < math.degrees(math.atan(D_abs / height_cam)):
        
        if in_angle <= (FOV_hor / 2 + 8):  
            return 'transform'
        
        elif in_angle > (FOV_hor / 2 + 8):
            return 'visTraj_del'
        
        if in_angle > (FOV_hor / 2 + 12):
            return 'ais_del'

def transform(AIS_current, AIS_vis, camera_para, shape):

    AIS_visCurrent = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','x','y','timestamp'])
    
    for index, ais in AIS_current.iterrows():

        flag = data_filter(ais, camera_para)
        if flag == 'transform':
            x, y = visual_transform(ais['lon'], ais['lat'], camera_para, shape)
            ais['x'], ais['y'] = x, y
            AIS_visCurrent = pd.concat([AIS_visCurrent, ais.to_frame().T], ignore_index=True)
        
        elif flag == 'visTraj_del' or flag == 'ais_del':
            AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['mmsi'] == ais['mmsi']].index)
    return AIS_vis, AIS_visCurrent

def data_pre(ais, timestamp):

    if ais['speed'] == 0:
        ais['timestamp'] = timestamp
    
    else:
        geo_d = pyproj.Geod(ellps="WGS84")
        
        distance = ais['speed'] * ((timestamp - ais['timestamp']) / 3600) * 1852
        ais['timestamp'] = timestamp

        ais['lon'], ais['lat'], c = geo_d.fwd(
            ais['lon'], ais['lat'], ais['course'], distance)
    return ais

def data_pred(AIS_cur, AIS_read, AIS_las, timestamp):

    # Time offset correction - AIS data is ~5 hours behind video
    TIME_OFFSET = 5 * 3600 * 1000  # 5 hours in milliseconds

    for index, ais in AIS_read.iterrows():
        # Apply time offset to AIS timestamp
        ais['timestamp'] = ais['timestamp'] + TIME_OFFSET
        ais['timestamp'] = round(ais['timestamp']/1000)
        
        if ais['timestamp'] == int(timestamp//1000):
            AIS_cur = pd.concat([AIS_cur, ais.to_frame().T], ignore_index=True)
        
        else:
            predicted_data = data_pre(ais, timestamp//1000)
            AIS_cur = pd.concat([AIS_cur, predicted_data.to_frame().T], ignore_index=True)
    
    for index, ais in AIS_las.iterrows():
        if ais['mmsi'] not in AIS_cur['mmsi'].values:
            predicted_data = data_pre(ais, timestamp//1000)
            AIS_cur = pd.concat([AIS_cur, predicted_data.to_frame().T], ignore_index=True)
    return AIS_cur

def data_coarse_process(AIS_current,AIS_last,camera_para,max_dis):

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
        dis = count_distance(camera_loc, ship_loc, Type='m')
        if dis > max_dis or data_filter(ais, camera_para) == 'ais_del':
            AIS_current = AIS_current.drop(index=index)
    return AIS_current

class AISPRO(object):
    def __init__(self, ais_path, ais_file, im_shape, t):
        
        self.ais_path = ais_path
        self.ais_file = ais_file
        self.im_shape = im_shape
        self.max_dis  = 2*1852
        self.t        = t
        self.time_lim = 2
        self.AIS_cur  = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','timestamp'])
        # self.AIS_row  = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','time'])
        # self.AIS_pre  = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','time'])
        self.AIS_vis  = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','x','y','timestamp'])
    
    def initialization(self):
        
        AIS_las = self.AIS_cur
        AIS_vis = self.AIS_vis
        AIS_cur = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','timestamp'])
        return AIS_cur, AIS_las, AIS_vis
    
    def read_ais(self, Time_name):

        try:
            path = self.ais_path + '/' + Time_name + '.csv'
            ais_data = pd.read_csv(path, usecols=[1, 2, 3, 4, 5, 6, 7, 8], header=0)
            # self.AIS_row = self.AIS_row.append(ais_data, ignore_index=True)
        except:
            ais_data = pd.DataFrame(columns=['mmsi','lon','lat','speed','course','heading','type','timestamp'])
        return ais_data
    
    def data_tran(self, AIS_cur, AIS_vis, camera_para, timestamp):

        AIS_vis, AIS_vis_cur = transform(AIS_cur, AIS_vis, camera_para, self.im_shape)

        # self.AIS_pre = self.AIS_pre.append(self.AIS_cur, ignore_index=True)
        AIS_vis = pd.concat([AIS_vis, AIS_vis_cur], ignore_index=True)
        
        # self.AIS_pre = self.AIS_pre.drop(self.AIS_pre[self.AIS_pre['time'] < (timestamp // 1000 - self.time_lim * 60)].index)
        AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['timestamp'] < (
                timestamp//1000 - self.time_lim * 60)].index)
        return AIS_vis
    
    def ais_pro(self, AIS_cur, AIS_las, AIS_vis, camera_para, timestamp, Time_name):

        AIS_read = self.read_ais(Time_name)      
        AIS_read = data_coarse_process(AIS_read, AIS_las,camera_para, self.max_dis)
        AIS_cur = data_pred(AIS_cur, AIS_read, AIS_las, timestamp)
        AIS_vis = self.data_tran(AIS_cur, AIS_vis,camera_para, timestamp)
        return AIS_vis, AIS_cur
    
    def process(self, camera_para, timestamp, Time_name):
        
        if timestamp % 1000 < self.t:
            Time_name = Time_name[:-4]
            AIS_cur, AIS_las, AIS_vis = self.initialization()
            self.AIS_vis, self.AIS_cur = self.ais_pro(AIS_cur,AIS_las, AIS_vis, camera_para, timestamp, Time_name)
        return self.AIS_vis, self.AIS_cur