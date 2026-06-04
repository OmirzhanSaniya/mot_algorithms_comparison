import json
import numpy as np
from common_m3ot import M3OTLoader, simulate_detector_from_gt, bbox_xywh_to_xy_center

JSON_PATH = r"D:\kbtu\6 semester\information theory\M3OT\M3OT\Annotations\1\ir\test_cocoformat.json"
VIDEO_ID = 9

loader = M3OTLoader(JSON_PATH)
sequence = loader.get_video_sequence(VIDEO_ID)

# GT из sequence
gt_per_frame = []
for frame in sequence:
    objs = []
    for det in frame["detections"]:
        z = bbox_xywh_to_xy_center(det["bbox_xywh"])
        objs.append({
            "track_id": det["gt_track_id"],
            "xy": z,
            "bbox_xywh": det["bbox_xywh"],
        })
    gt_per_frame.append(objs)

# "Испорченные" detections
detections_per_frame, _ = simulate_detector_from_gt(
    sequence,
    meas_noise_std=3.0,
    miss_prob=0.2,
    false_per_frame=2,
    image_size=(640, 512),
    seed=42,
)

#print("\n=== FIRST 3 FRAMES COMPARISON ===\n")

#for i in range(min(3, len(sequence))):
    #print(f"\nFRAME {i+1}")
    #print("file_name:", sequence[i]["file_name"])

    #print("\nGT objects:")
    #for obj in gt_per_frame[i]:
     #   print({
         #   "track_id": obj["track_id"],
          #  "bbox_xywh": obj["bbox_xywh"],
           # "center_xy": obj["xy"].tolist()
      #  })

    #print("\nNoisy detections:")
    #for det in detections_per_frame[i]:
    #    print(det.tolist())

# ----------------------------
# Save noisy detections
# ----------------------------
def save_full_debug(sequence, gt_per_frame, detections_per_frame, path="debug_output.json"):
    data = []

    for i in range(len(sequence)):
        frame_data = {
            "frame": i + 1,
            "file_name": sequence[i]["file_name"],
            "gt": [],
            "noisy_detections": []
        }

        # GT
        for obj in gt_per_frame[i]:
            frame_data["gt"].append({
                "track_id": int(obj["track_id"]),
                "bbox_xywh": [float(v) for v in obj["bbox_xywh"]],
                "center_xy": [float(v) for v in obj["xy"]]
            })

        # noisy detections
        for det in detections_per_frame[i]:
            frame_data["noisy_detections"].append([
                float(det[0]),
                float(det[1])
            ])

        data.append(frame_data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

save_full_debug(sequence, gt_per_frame, detections_per_frame)
print("\nSaved debug_output.json")