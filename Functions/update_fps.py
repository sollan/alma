#!/usr/bin/env python3
"""
Command-line utility to update FPS in config.yaml from a video file
Usage: python Functions/update_fps.py <video_file>
       or: cd Functions && python update_fps.py <video_file>
"""

import sys
import os

# Handle imports for both standalone execution and module import
try:
    from . import ConfigFunctions
except ImportError:
    # If running as standalone script, add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Functions import ConfigFunctions

def main():
    if len(sys.argv) < 2:
        print("Usage: python Functions/update_fps.py <video_file>")
        print("   or: cd Functions && python update_fps.py <video_file>")
        print("\nExample:")
        print("  python Functions/update_fps.py my_video.mp4")
        print("\nSupported formats: mp4, avi, mov, mkv")
        sys.exit(1)
    
    video_path = sys.argv[1]
    config_path = './config.yaml'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Make sure you're running this from the ALMA directory")
        sys.exit(1)
    
    print(f"Reading FPS from: {video_path}")
    print("-" * 50)
    
    fps = ConfigFunctions.update_fps_from_video(config_path, video_path)
    
    if fps is not None:
        print("-" * 50)
        print(f"✓ SUCCESS: Config updated with FPS = {fps}")
        print(f"\nYou can now run ALMA with this frame rate.")
    else:
        print("-" * 50)
        print("✗ FAILED: Could not update config")
        sys.exit(1)

if __name__ == '__main__':
    main()

