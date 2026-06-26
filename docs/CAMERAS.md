# Connect your camera 📷

EveryCam works with almost any camera you already own. Faces and license plates are blurred
automatically **before** anything is saved. Pick your device below.

> On macOS, the first time you use a live camera you may need to allow your terminal app under
> **System Settings → Privacy & Security → Camera**.

| Platform | How to connect | Command |
|---|---|---|
| **Laptop / USB webcam** | built-in or plugged-in camera | `everycam capture --preset webcam` (use `--device 1` for a second camera) |
| **Phone — recorded** | record a clip, copy the `.mp4` to your computer | `everycam capture --preset phone --path clip.mp4` |
| **Phone — wireless/live** | install an "IP Webcam" (Android) / similar app; it shows a URL like `http://192.168.0.5:8080/video` | `everycam capture --preset phone_ip --path http://192.168.0.5:8080/video` |
| **Dashcam** | copy the `.mp4` from the SD card | `everycam capture --preset dashcam --path drive.mp4` |
| **IP / CCTV / RTSP camera** | get the camera's RTSP URL (`rtsp://user:pass@192.168.0.9:554/stream`) | `everycam capture --preset ipcam --path rtsp://user:pass@192.168.0.9:554/stream` |
| **Smart glasses** (Ray-Ban Meta, Aria, Vision Pro) | export the recorded video to a file | `everycam capture --preset glasses --path clip.mp4` |
| **Action cam** (GoPro, Insta360) | copy the `.mp4`, or use the cam's webcam/UVC mode | `everycam capture --preset gopro --path clip.mp4` |
| **A folder of image frames** | point at the folder | `everycam capture --preset frames --path my_frames/` |

## Tips for good clips

- Keep the **hands and the object** clearly in view, steady, and well-lit.
- 10–60 seconds is plenty. EveryCam grabs ~150 frames by default (`--max-frames` to change).
- For the best hand tracking, install MediaPipe (`pip install -e ".[hands]"`) and add
  `--hands mediapipe`. Without it, a dependency-free fallback is used.

## What happens to my video?

The privacy gate blurs faces and plates on every frame **before** storage. If you contribute the
"signals only" way, only the *numbers* (where the hand moved) leave your machine — never the video
of people. See [PRIVACY.md](../PRIVACY.md).
