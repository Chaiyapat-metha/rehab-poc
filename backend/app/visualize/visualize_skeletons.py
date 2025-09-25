# rehab-poc/backend/app/visualize/visualize_skeletons.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import psycopg2
import os
import sys

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import db_ops for database connection
from backend.app.utils import db

# --- Mediapipe Connections for 3D Plotting ---
# These are the pairs of joints that form a connection (e.g., bone)
connections = [
    (11, 12),  # Shoulders
    (12, 14), (14, 16), # Right arm
    (11, 13), (13, 15), # Left arm
    (12, 24), (24, 26), (26, 28), (28, 30), (30, 32), # Right leg
    (11, 23), (23, 25), (25, 27), (27, 29), (29, 31), # Left leg
    (23, 24) # Hips
]

def get_skeleton_data(video_id: str) -> np.ndarray:
    """
    Fetches raw keypoint data for a specific video_id from the database.
    
    Args:
        video_id (str): The unique ID of the video session to fetch.
        
    Returns:
        np.ndarray: A NumPy array of shape (num_frames, 33, 3).
    """
    sql = """
        SELECT keypoints
        FROM training_skeletons
        WHERE video_id = %s
        ORDER BY frame_no ASC;
    """
    try:
        results = db.execute_query(sql, (video_id,))
        if not results:
            print(f"Error: No data found for video ID: {video_id}")
            return None
        
        # Flatten the list of lists into a single NumPy array
        flat_data = np.array([item[0] for item in results], dtype=np.float32)
        
        # Reshape to (num_frames, 33, 3)
        num_frames = len(results)
        keypoints_3d = flat_data.reshape(num_frames, 33, 3)
        
        print(f"✅ Successfully fetched {num_frames} frames from video ID: {video_id}")
        return keypoints_3d

    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

def plot_time_series(keypoints: np.ndarray, label: str):
    """
    Plots the time-series of X, Y, Z coordinates for all joints.
    """
    print(f"-> Plotting time series for '{label}' video...")
    num_frames = keypoints.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Define a list of colors for better visualization
    colors = plt.cm.get_cmap('tab20', 33)
    
    # Plot X, Y, Z for all joints
    for i in range(33):
        # The fix: Add a label for each line
        axes[0].plot(keypoints[:, i, 0], alpha=0.5, color=colors(i), label=f"Joint {i}") # X
        axes[1].plot(keypoints[:, i, 1], alpha=0.5, color=colors(i), label=f"Joint {i}") # Y
        axes[2].plot(keypoints[:, i, 2], alpha=0.5, color=colors(i), label=f"Joint {i}") # Z
    
    axes[0].set_title('X Coordinate Over Time')
    axes[1].set_title('Y Coordinate Over Time')
    axes[2].set_title('Z Coordinate Over Time')
    axes[2].set_xlabel('Frame Number')
    
    fig.suptitle(f"Time-Series Plot for '{label}' video", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # The fix: Add the legend to show joint labels
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    axes[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    plt.show()
    
def animate_3d_skeleton(keypoints: np.ndarray, label: str):
    """
    Creates an animated 3D plot of the skeleton.
    
    Args:
        keypoints (np.ndarray): Shape (num_frames, 33, 3).
        label (str): The label of the video (e.g., 'correct').
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set plot limits for consistency
    all_coords = keypoints.reshape(-1, 3)
    ax.set_xlim([all_coords[:, 0].min(), all_coords[:, 0].max()])
    ax.set_ylim([all_coords[:, 1].min(), all_coords[:, 1].max()])
    ax.set_zlim([all_coords[:, 2].min(), all_coords[:, 2].max()])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Skeleton Animation for '{label}' video")
    
    # Create empty plot objects
    points, = ax.plot([], [], [], 'o', color='blue')
    lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]

    def update(frame_idx):
        frame_data = keypoints[frame_idx]
        
        # Update point coordinates
        points._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        
        # Update bone lines
        for i, connection in enumerate(connections):
            p1_idx, p2_idx = connection
            p1 = frame_data[p1_idx]
            p2 = frame_data[p2_idx]
            lines[i].set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
            
        return points, *lines

    num_frames = keypoints.shape[0]
    ani = FuncAnimation(fig, update, frames=range(num_frames), interval=50, blit=True)
    
    # Save the animation as a video file (optional)
    # You may need to install ffmpeg first: `conda install -c conda-forge ffmpeg`
    # ani.save(f'3d_skeleton_{label}.mp4', writer='ffmpeg', fps=20)
    
    plt.show()

def main():
    """Main function to visualize a specific video."""
    video_id = 'f439bad7-91a7-4918-a46c-ee6deac9d2f6'
    label = 'correct'
    
    keypoints_data = get_skeleton_data(video_id)
    if keypoints_data is not None:
        plot_time_series(keypoints_data, label)
        animate_3d_skeleton(keypoints_data, label) # Call the animation function

if __name__ == '__main__':
    main()