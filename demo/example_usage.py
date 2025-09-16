"""
DeepSORVF Vessel Tracking System - Usage Examples
Simple examples showing how to use the extracted modules
"""

import pandas as pd
import numpy as np
import cv2
import os

# Import the unified tracker
try:
    from vessel_tracker import VesselTracker
    print("✅ Unified tracker available")
except ImportError as e:
    print(f"❌ Unified tracker import error: {e}")

# Import individual modules for testing
try:
    from ais import AISPRO
    print("✅ AIS module available")
    AIS_AVAILABLE = True
except ImportError as e:
    print(f"❌ AIS module import error: {e}")
    AIS_AVAILABLE = False

try:
    from vis import VISPRO
    print("✅ VIS module available")
    VIS_AVAILABLE = True
except ImportError as e:
    print(f"❌ VIS module import error: {e}")
    VIS_AVAILABLE = False

try:
    from fusion import Fusion
    print("✅ Fusion module available")
    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Fusion module import error: {e}")
    FUSION_AVAILABLE = False


def example_1_unified_tracker():
    """Complete vessel tracking with all modules"""
    print("\n🚢 Example 1: Unified Vessel Tracking")
    
    # Configuration
    config = {
        'ais_config': {
            'data_path': './clip-01/ais',
            'files': []  # Auto-detect CSV files
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
            'parameters': [114.327, 30.600, 45.0, 1500, 50, 0.02, 0.65]
        }
    }
    
    try:
        # Initialize tracker
        tracker = VesselTracker(**config)
        
        # Create dummy frame for testing
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        timestamp = 1654336512000  # Example timestamp
        time_name = "2022_06_04_12_05_12"
        
        # Process frame
        results = tracker.process_frame(test_frame, timestamp, time_name)
        
        # Print results
        print(f"  📊 Processing Results:")
        print(f"    AIS vessels: {len(results['ais']['current'])}")
        print(f"    VIS tracks: {len(results['vis']['current'])}")
        print(f"    Fused data: {len(results['fusion']['matched'])}")
        print(f"    Processing time: {results['metadata']['processing_time']:.2f}ms")
        
        # Get status
        status = tracker.get_vessel_status()
        print(f"  🎯 Status: {status['total_count']} total vessels")
        
        # Performance metrics
        metrics = tracker.get_performance_metrics()
        print(f"  ⚡ Performance: {metrics['fps_capability']:.1f} FPS capability")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def example_2_ais_only():
    """AIS processing only"""
    print("\n📡 Example 2: AIS Processing Only")
    
    if not AIS_AVAILABLE:
        print("  ❌ AIS module not available")
        return False
    
    try:
        # Initialize AIS processor with original interface
        ais = AISPRO(
            ais_path='../clip-01/ais',
            ais_file=[],  # Not used in original
            im_shape=[1920, 1080],
            t=33
        )
        
        # Camera parameters
        camera_params = [114.327, 30.600, 45.0, 1500, 50, 0.02, 0.65]
        timestamp = 1654336512000
        time_name = "2022_06_04_12_05_12"
        
        # Process AIS data
        ais_visible, ais_current = ais.process(camera_params, timestamp, time_name)
        
        print(f"  📊 Results:")
        print(f"    Visible vessels: {len(ais_visible)}")
        print(f"    Current vessels: {len(ais_current)}")
        
        if not ais_visible.empty:
            print(f"    Sample vessel MMSI: {ais_visible.iloc[0]['mmsi'] if 'mmsi' in ais_visible.columns else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def example_3_vis_only():
    """Visual tracking only"""
    print("\n👁️  Example 3: Visual Tracking Only")
    
    if not VIS_AVAILABLE:
        print("  ❌ VIS module not available")
        return False
    
    try:
        # Initialize VIS processor with original interface
        vis = VISPRO(
            anti=1,    # anti-occlusion enabled
            val=0.3,   # occlusion threshold
            t=33       # time interval
        )
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        timestamp = 1654336512000
        
        # Empty AIS and binding data for VIS-only mode
        ais_data = pd.DataFrame()
        bin_inf = pd.DataFrame()
        
        # Process frame with original interface
        vis_tra, vis_cur = vis.feedCap(test_frame, timestamp, ais_data, bin_inf)
        
        print(f"  📊 Results:")
        print(f"    Trajectory points: {len(vis_tra)}")
        print(f"    Current detections: {len(vis_cur)}")
        
        if not vis_cur.empty:
            print(f"    Sample detection ID: {vis_cur.iloc[0]['ID'] if 'ID' in vis_cur.columns else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def example_4_fusion_only():
    """Sensor fusion only"""
    print("\n🔗 Example 4: Sensor Fusion Only")
    
    if not FUSION_AVAILABLE:
        print("  ❌ Fusion module not available")
        return False
    
    try:
        # Initialize fusion processor
        fusion = Fusion(
            max_distance=200,
            image_shape=[1920, 1080],
            time_interval=33
        )
        
        # Create sample AIS data (matching expected format)
        ais_visible = pd.DataFrame({
            'mmsi': [123456, 789012],
            'lon': [28.9765, 28.9823],
            'lat': [41.0082, 41.0156],
            'speed': [12.5, 8.3],
            'course': [45.2, 180.0],
            'heading': [50.1, 185.2],
            'type': [70, 36],
            'x': [500.0, 800.0],
            'y': [400.0, 600.0],
            'timestamp': [1654336512000, 1654336512000]
        })
        
        # Create sample VIS data (matching expected format)
        vis_trajectories = pd.DataFrame({
            'ID': [1, 2],
            'bbox_x': [495.0, 785.0],
            'bbox_y': [395.0, 590.0],
            'bbox_w': [30.0, 35.0],
            'bbox_h': [40.0, 45.0],
            'x': [510.0, 790.0],
            'y': [405.0, 595.0],
            'timestamp': [1654336512000, 1654336512000]
        })
        
        # Current data (with matching vessels)
        ais_current = pd.DataFrame({
            'mmsi': [123456, 789012],
            'lon': [28.9765, 28.9823],
            'lat': [41.0082, 41.0156],
            'speed': [12.5, 8.3],
            'course': [45.2, 180.0],
            'heading': [50.1, 185.2],
            'type': [70, 36],
            'timestamp': [1654336512000, 1654336512000]
        })
        
        vis_current = pd.DataFrame({
            'ID': [1, 2],
            'bbox_x': [495.0, 785.0],
            'bbox_y': [395.0, 590.0],
            'bbox_w': [30.0, 35.0],
            'bbox_h': [40.0, 45.0],
            'timestamp': [1654336512000, 1654336512000]
        })
        
        # Perform fusion
        fused_data, bin_inf = fusion.fusion(
            ais_visible, ais_current, vis_trajectories, vis_current, 1654336512000
        )
        
        print(f"  📊 Results:")
        print(f"    Input AIS vessels: {len(ais_visible)}")
        print(f"    Input VIS tracks: {len(vis_trajectories)}")
        print(f"    Fused matches: {len(fused_data)}")
        print(f"    Binding records: {len(bin_inf)}")
        
        if not fused_data.empty:
            print(f"    Sample match: MMSI {fused_data.iloc[0]['mmsi']} ↔ ID {fused_data.iloc[0]['ID']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def example_5_video_processing():
    """Process a video file (if available)"""
    print("\n🎥 Example 5: Video Processing")
    
    video_path = './clip-01/2022_06_04_12_05_12_12_07_02_b.mp4'
    
    if not os.path.exists(video_path):
        print(f"  ❌ Video file not found: {video_path}")
        return False
    
    try:
        # Configuration for video processing
        config = {
            'ais_config': {
                'data_path': './clip-01/ais',
                'files': []
            },
            'vis_config': {
                'anti_occlusion': False,  # Disable for speed
                'detection_model': 'simple'
            },
            'fusion_config': {
                'max_distance': 150,
                'time_interval': 33
            },
            'camera_config': {
                'image_shape': [1920, 1080],
                'parameters': [114.327, 30.600, 45.0]
            }
        }
        
        # Initialize tracker
        tracker = VesselTracker(**config)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        max_frames = 10  # Process only first 10 frames for demo
        
        print(f"  📹 Processing {max_frames} frames...")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get timestamp
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Process frame
            results = tracker.process_frame(frame, timestamp)
            
            # Print progress
            if frame_count % 5 == 0:
                ais_count = len(results['ais']['current'])
                vis_count = len(results['vis']['current'])
                fused_count = len(results['fusion']['matched'])
                proc_time = results['metadata']['processing_time']
                
                print(f"    Frame {frame_count}: AIS={ais_count}, VIS={vis_count}, Fused={fused_count}, Time={proc_time:.1f}ms")
            
            frame_count += 1
        
        cap.release()
        
        # Final statistics
        metrics = tracker.get_performance_metrics()
        print(f"  📊 Final Stats:")
        print(f"    Processed {frame_count} frames")
        print(f"    Average FPS capability: {metrics['fps_capability']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def example_6_data_analysis():
    """Analyze available data files"""
    print("\n📊 Example 6: Data Analysis")
    
    # Check AIS data
    ais_path = './clip-01/ais'
    if os.path.exists(ais_path):
        ais_files = [f for f in os.listdir(ais_path) if f.endswith('.csv')]
        print(f"  📡 AIS Data:")
        print(f"    Path: {ais_path}")
        print(f"    Files: {len(ais_files)} CSV files")
        
        if ais_files:
            # Analyze first file
            sample_file = os.path.join(ais_path, ais_files[0])
            try:
                df = pd.read_csv(sample_file)
                print(f"    Sample file: {ais_files[0]}")
                print(f"    Columns: {list(df.columns)}")
                print(f"    Records: {len(df)}")
                
                if 'mmsi' in df.columns:
                    unique_vessels = df['mmsi'].nunique()
                    print(f"    Unique vessels: {unique_vessels}")
                    
            except Exception as e:
                print(f"    Error reading sample: {e}")
    else:
        print(f"  ❌ AIS data not found at {ais_path}")
    
    # Check camera parameters
    camera_file = './clip-01/camera_para.txt'
    if os.path.exists(camera_file):
        print(f"  📷 Camera Parameters:")
        try:
            with open(camera_file, 'r') as f:
                params = f.read().strip()
            print(f"    File: {camera_file}")
            print(f"    Content: {params[:100]}...")  # First 100 chars
        except Exception as e:
            print(f"    Error reading camera params: {e}")
    else:
        print(f"  ❌ Camera parameters not found at {camera_file}")
    
    # Check video file
    video_file = './clip-01/2022_06_04_12_05_12_12_07_02_b.mp4'
    if os.path.exists(video_file):
        print(f"  🎥 Video File:")
        print(f"    File: {video_file}")
        file_size = os.path.getsize(video_file) / (1024*1024)  # MB
        print(f"    Size: {file_size:.1f} MB")
        
        try:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"    Resolution: {width}x{height}")
            print(f"    FPS: {fps:.1f}")
            print(f"    Duration: {duration:.1f} seconds")
            print(f"    Frames: {frame_count}")
            
            cap.release()
        except Exception as e:
            print(f"    Error analyzing video: {e}")
    else:
        print(f"  ❌ Video file not found at {video_file}")


if __name__ == "__main__":
    print("🚢 DeepSORVF Vessel Tracking System - Examples")
    print("=" * 50)
    
    # Run all examples
    examples = [
        example_6_data_analysis,    # Data analysis first
        example_2_ais_only,        # Individual modules
        example_3_vis_only,
        example_4_fusion_only,
        example_1_unified_tracker, # Unified system
        example_5_video_processing # Video processing last
    ]
    
    results = []
    for example in examples:
        try:
            success = example()
            if success is None:
                success = True  # Treat None as success
            results.append(success)
        except Exception as e:
            print(f"Example failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    successful = sum(results)
    total = len(results)
    print(f"  ✅ {successful}/{total} examples completed successfully")
    
    if successful == total:
        print("  🎉 All systems operational!")
    elif successful > 0:
        print("  ⚠️  Partial functionality - check dependencies")
    else:
        print("  ❌ No examples successful - check installation")