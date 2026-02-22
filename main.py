from face_roi import extract_roi_from_video
from rppg_signal import RPPGSignal
from hrv_features import HRVFeatures

# ↓↓↓ PUT YOUR VIDEO FILENAME HERE ↓↓↓
VIDEO_PATH = r"C:\Users\vibha\OneDrive\Desktop\heart\vid.avi"
# ↑↑↑ Change "my_video.mp4" to your actual video file name ↑↑↑

print("Step 1: Extracting RGB from video... (this may take a moment)")
rgb_means = extract_roi_from_video(VIDEO_PATH)
print(f"  Done! Extracted {rgb_means.shape[0]} frames.")

print("Step 2: Processing rPPG signal...")
rppg = RPPGSignal(fps=30)  # Change 30 if your video has different FPS
waveform = rppg.process(rgb_means)
print("  Done!")

print("Step 3: Computing HRV features...")
hrv = HRVFeatures(fps=30)  # Change 30 if your video has different FPS
results = hrv.compute(waveform)

print("\n========== RESULTS ==========")
print(f"Heart Rate : {round(results['heart_rate_bpm'], 2)} BPM")
print(f"SDNN       : {round(results['sdnn_ms'], 2)} ms")
print(f"RMSSD      : {round(results['rmssd_ms'], 2)} ms")
print(f"Num Beats  : {results['num_beats']}")
print("==============================")