"""
Unified Vessel Tracking System
Combines AIS, VIS, and FUS modules for complete vessel tracking

Usage:
    from vessel_tracker import VesselTracker
    
    tracker = VesselTracker(
        ais_config={
            'data_path': '/path/to/ais',
            'files': ['file1.csv', 'file2.csv']
        },
        vis_config={
            'anti_occlusion': True,
            'detection_model': 'yolox'
        },
        fusion_config={
            'max_distance': 200,
            'time_interval': 33
        },
        camera_config={
            'image_shape': [1920, 1080],
            'parameters': [lon, lat, heading, ...]
        }
    )
    
    # Process single frame
    results = tracker.process_frame(frame, timestamp, time_name)
    
    # Get tracking results
    ais_data = results['ais']
    vis_data = results['vis'] 
    fused_data = results['fusion']
"""

import pandas as pd
import numpy as np
import warnings
import cv2
warnings.filterwarnings('ignore')

try:
    from ais_realtime import AISPRO
    AIS_AVAILABLE = True
except ImportError:
    print("Warning: AISPRO module not available")
    AIS_AVAILABLE = False

try:
    from vis import VISPRO
    VIS_AVAILABLE = True
except ImportError:
    print("Warning: VISPRO module not available")
    VIS_AVAILABLE = False

try:
    from fusion import Fusion
    FUSION_AVAILABLE = True
except ImportError:
    print("Warning: Fusion module not available")
    FUSION_AVAILABLE = False


class VesselTracker:
    def __init__(self, ais_config=None, vis_config=None, fusion_config=None, camera_config=None):
        """
        Unified Vessel Tracking System
        
        Args:
            ais_config (dict): AIS configuration
            vis_config (dict): VIS configuration  
            fusion_config (dict): Fusion configuration
            camera_config (dict): Camera configuration
        """
        self.camera_config = camera_config or {}
        self.image_shape = self.camera_config.get('image_shape', [1920, 1080])
        self.camera_parameters = self.camera_config.get('parameters', [])
        
        # VesselTracker.__init__ içinde
        if AIS_AVAILABLE and ais_config:
            # Realtime arayüz
            self.ais_processor = AISPRO(
                ais_host=ais_config.get('host', '127.0.0.1'),
                ais_port=ais_config.get('port', 10110),
                im_shape=self.image_shape,
                t=(fusion_config.get('time_interval', 33) if fusion_config else 33)
            )
        else:
            self.ais_processor = None
            print("AIS processor not initialized")

        # Initialize VIS processor with original interface
        if VIS_AVAILABLE and vis_config:
            self.vis_processor = VISPRO(
                anti=1 if vis_config.get('anti_occlusion', True) else 0,
                val=vis_config.get('occlusion_rate', 0.3),
                t=fusion_config.get('time_interval', 33) if fusion_config else 33
            )
        else:
            self.vis_processor = None
            print("VIS processor not initialized")
        
        # Initialize Fusion processor
        if FUSION_AVAILABLE and fusion_config:
            self.fusion_processor = Fusion(
                max_distance=fusion_config.get('max_distance', 200),
                image_shape=self.image_shape,
                time_interval=fusion_config.get('time_interval', 33)
            )
        else:
            self.fusion_processor = None
            print("Fusion processor not initialized")
        
        # Initialize tracking state
        self.bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        self.last_results = {}
    
    def process_frame(self, frame, timestamp, time_name=None):
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input image frame (numpy array)
            timestamp: Current timestamp (milliseconds)
            time_name: Time string for AIS file lookup (optional)
            
        Returns:
            dict: Processing results containing AIS, VIS, and fusion data
        """
        results = {
            'ais': {'visible': pd.DataFrame(), 'current': pd.DataFrame()},
            'vis': {'trajectories': pd.DataFrame(), 'current': pd.DataFrame()},
            'fusion': {'matched': pd.DataFrame(), 'bindings': pd.DataFrame()},
            'metadata': {
                'timestamp': timestamp,
                'time_name': time_name,
                'processing_time': 0
            }
        }
        
        start_time = pd.Timestamp.now()
        
        try:
            # Step 1: Process AIS data
            if self.ais_processor and self.camera_parameters:
                # Yeni (realtime)
                ais_vis, ais_cur = self.ais_processor.process(self.camera_parameters, timestamp)

                results['ais']['visible'] = ais_vis
                results['ais']['current'] = ais_cur
            else:
                ais_vis = pd.DataFrame()
                ais_cur = pd.DataFrame()
            
            # Step 2: Process VIS data with original interface
            if self.vis_processor is not None and frame is not None:
                vis_tra, vis_cur = self.vis_processor.feedCap(
                    frame, timestamp, ais_vis, self.bin_inf
                )
                results['vis']['trajectories'] = vis_tra
                results['vis']['current'] = vis_cur
            else:
                vis_tra = pd.DataFrame()
                vis_cur = pd.DataFrame()
            
            # Step 3: Fusion processing
            if self.fusion_processor and not ais_vis.empty and not vis_tra.empty:
                fused_data, self.bin_inf = self.fusion_processor.fusion(
                    ais_vis, ais_cur, vis_tra, vis_cur, timestamp
                )
                results['fusion']['matched'] = fused_data
                results['fusion']['bindings'] = self.bin_inf
            else:
                results['fusion']['matched'] = pd.DataFrame()
                results['fusion']['bindings'] = self.bin_inf
            
            for _, vessel in results['ais']['current'].iterrows():
                mmsi = vessel['mmsi']
                sog = vessel['speed']
                cog = vessel['course']
                lat = vessel['lat']
                lon = vessel['lon']

                # Piksel koordinatına çevir (AISPRO'nun visual_transform'u ile)
                try:
                    x, y = AISPRO.visual_transform(lon, lat, self.camera_parameters, self.image_shape)
                    cv2.putText(frame, f"MMSI: {mmsi}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.putText(frame, f"SOG: {sog}  COG: {cog}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.putText(frame, f"LAT: {lat:.5f}  LON: {lon:.5f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                except Exception as e:
                    pass

            # Calculate processing time
            end_time = pd.Timestamp.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            results['metadata']['processing_time'] = processing_time
            
            # Store results for access
            self.last_results = results
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            results['metadata']['error'] = str(e)
        
        return results, frame
    
    def get_vessel_status(self):
        """
        Get current status of all tracked vessels
        
        Returns:
            dict: Vessel status information
        """
        status = {
            'ais_vessels': [],
            'vis_tracks': [],
            'fused_vessels': [],
            'total_count': 0
        }
        
        if not self.last_results:
            return status
        
        # AIS vessels
        ais_current = self.last_results['ais']['current']
        if not ais_current.empty:
            status['ais_vessels'] = ais_current['mmsi'].tolist()
        
        # VIS tracks
        vis_current = self.last_results['vis']['current']
        if not vis_current.empty:
            status['vis_tracks'] = vis_current['ID'].tolist()
        
        # Fused vessels
        fused_data = self.last_results['fusion']['matched']
        if not fused_data.empty:
            status['fused_vessels'] = [
                {'ais_mmsi': row['mmsi'], 'vis_id': row['ID']} 
                for _, row in fused_data.iterrows()
            ]
        
        status['total_count'] = len(set(status['ais_vessels'] + status['vis_tracks']))
        
        return status
    
    def get_performance_metrics(self):
        """
        Get performance metrics
        
        Returns:
            dict: Performance information
        """
        if not self.last_results:
            return {}
        
        metadata = self.last_results['metadata']
        
        metrics = {
            'processing_time_ms': metadata.get('processing_time', 0),
            'fps_capability': 1000 / max(metadata.get('processing_time', 1), 1),
            'ais_count': len(self.last_results['ais']['current']),
            'vis_count': len(self.last_results['vis']['current']),
            'fusion_count': len(self.last_results['fusion']['matched']),
            'binding_count': len(self.last_results['fusion']['bindings'])
        }
        
        return metrics
    
    def reset_tracking(self):
        """Reset all tracking states"""
        if self.ais_processor:
            self.ais_processor.reset()
        if self.vis_processor:
            self.vis_processor.reset()
        if self.fusion_processor:
            self.fusion_processor.reset()
        
        self.bin_inf = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        self.last_results = {}
    
    def save_results(self, filepath, format='csv'):
        """
        Save tracking results to file
        
        Args:
            filepath (str): Output file path
            format (str): Output format ('csv', 'json')
        """
        if not self.last_results:
            print("No results to save")
            return
        
        try:
            if format == 'csv':
                # Save fusion results as main output
                fusion_data = self.last_results['fusion']['matched']
                if not fusion_data.empty:
                    fusion_data.to_csv(filepath, index=False)
                    print(f"Results saved to {filepath}")
                else:
                    print("No fusion data to save")
            
            elif format == 'json':
                # Convert dataframes to dict for JSON serialization
                output_data = {}
                for module, data in self.last_results.items():
                    if module != 'metadata':
                        output_data[module] = {}
                        for key, df in data.items():
                            if isinstance(df, pd.DataFrame):
                                output_data[module][key] = df.to_dict('records')
                            else:
                                output_data[module][key] = df
                    else:
                        output_data[module] = data
                
                import json
                with open(filepath, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(f"Results saved to {filepath}")
        
        except Exception as e:
            print(f"Error saving results: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("🚢 Unified Vessel Tracking System Test")
    
    # Configuration
    config = {
        'ais_config': {
            'host': '127.0.0.1',
            'port': 10110
        },
        'vis_config': {
            'anti_occlusion': True,
            'occlusion_rate': 0.3,
            'detection_model': 'yolox'
        },
        'fusion_config': {
            'max_distance': 200,
            'time_interval': 33
        },
        'camera_config': {
            'image_shape': [1920, 1080],
            'parameters': [114.327, 30.600, 45.0]  # lon, lat, heading
        }
    }
    
    # Initialize tracker
    tracker = VesselTracker(**config)
    
    # Test with dummy data
    try:
        # Create dummy frame
        import numpy as np
        cap = cv2.VideoCapture("2022_06_04_12_05_12_12_07_02_b.mp4")  # USB kamera
        # cap = cv2.VideoCapture("video.mp4")  # Video dosyası
        # cap = cv2.VideoCapture("rtsp://kullanici:sifre@kamera_ip:554/stream")  # IP kamera

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Zaman damgası (ms cinsinden)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            time_name = None  # Gerekirse video adı veya zaman etiketi

            # Tracker ile işle
            results, overlay_frame = tracker.process_frame(frame, timestamp, time_name)

            # Overlay göster
            cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Overlay", 1280, 720)
            cv2.imshow("Overlay", overlay_frame)

            # 'q' ile çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print results
        print("\n📊 Processing Results:")
        print(f"  AIS vessels: {len(results['ais']['current'])}")
        print(f"  VIS tracks: {len(results['vis']['current'])}")
        print(f"  Fused data: {len(results['fusion']['matched'])}")
        print(f"  Processing time: {results['metadata']['processing_time']:.2f}ms")
        
        # Print status
        status = tracker.get_vessel_status()
        print(f"\n🎯 Vessel Status:")
        print(f"  Total vessels: {status['total_count']}")
        print(f"  AIS vessels: {status['ais_vessels']}")
        print(f"  VIS tracks: {status['vis_tracks']}")
        print(f"  Fused vessels: {len(status['fused_vessels'])}")
        
        # Print performance
        metrics = tracker.get_performance_metrics()
        print(f"\n⚡ Performance:")
        print(f"  FPS capability: {metrics['fps_capability']:.1f}")
        print(f"  Processing time: {metrics['processing_time_ms']:.2f}ms")
        
    except Exception as e:
        print(f"Error in test: {e}")
        print("Note: Full functionality requires all dependencies")