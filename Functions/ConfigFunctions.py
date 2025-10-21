import yaml
import cv2
import os

def load_config(config_path):

    config_file = open(config_path)
    parsed_config_file = yaml.load(config_file, Loader=yaml.FullLoader)

    return parsed_config_file


def get_video_fps(video_path):
    '''
    Extract FPS from video file using OpenCV
    Returns: frame_rate (float) or None if error
    '''
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        print(f"Video metadata:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {duration:.2f}s")
        
        return fps
    
    except Exception as e:
        print(f"Error reading video: {e}")
        return None


def update_config(config_path, key, value):
    '''
    Update a specific key in the config.yaml file
    '''
    try:
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Update value
        config[key] = value
        
        # Write back with comments preserved
        # Read original file to preserve comments
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Find and update the line
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{key}:'):
                # Preserve indentation and comment
                indent = len(line) - len(line.lstrip())
                comment = ''
                if '#' in line:
                    comment = '  ' + line.split('#', 1)[1].rstrip()
                lines[i] = ' ' * indent + f'{key}: {value}{comment}\n'
                updated = True
                break
        
        if updated:
            # Write back
            with open(config_path, 'w') as f:
                f.writelines(lines)
            print(f"Config updated: {key} = {value}")
            return True
        else:
            print(f"Warning: Key '{key}' not found in config file")
            return False
    
    except Exception as e:
        print(f"Error updating config: {e}")
        return False


def update_fps_from_video(config_path, video_path):
    '''
    Read FPS from video and update config.yaml
    Returns: fps value or None
    '''
    fps = get_video_fps(video_path)
    
    if fps is not None and fps > 0:
        # Round to 2 decimal places for cleaner config
        fps_rounded = round(fps, 2)
        
        if update_config(config_path, 'frame_rate', fps_rounded):
            print(f"✓ Config updated with FPS: {fps_rounded}")
            return fps_rounded
    else:
        print("✗ Could not update config - invalid FPS")
        return None