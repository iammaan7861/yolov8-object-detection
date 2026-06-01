# 🚀 Quick Start Guide

## Get Running in 5 Minutes

### Step 1: Install Python
Ensure Python 3.8+ is installed. Check with:
```bash
python --version
```

### Step 2: Setup Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will automatically download YOLOv8 models on first run (50-100 MB).

### Step 4: Run the App
```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501` 🎉

---

## 📖 Basic Workflow

### 1️⃣ Detect Objects in Image
1. Click **📷 Image Detection** tab
2. Upload an image (JPG, PNG, etc.)
3. Click **🚀 Detect Objects**
4. See results with boxes and statistics

### 2️⃣ Analyze Video
1. Click **🎬 Video Detection** tab
2. Upload a video (MP4, AVI, etc.)
3. Click **🚀 Process Video**
4. Download the annotated video
5. View detailed statistics

### 3️⃣ Use Webcam
1. Click **📹 Live Webcam** tab
2. Click **📹 Start Webcam Detection**
3. Allow camera access
4. See live detections

### 4️⃣ Check Analytics
1. Click **📊 Analytics Dashboard** tab
2. See all detection history and statistics
3. Track trends across all uploads

---

## ⚙️ Adjust Settings in Sidebar

- **Model**: Choose yolov8n/s/m
  - nano = fastest ⚡
  - small = balanced ⚙️
  - medium = most accurate 🎯

- **Confidence Threshold**: 0.0-1.0
  - Lower = more detections
  - Higher = only confident detections

- **Advanced Options**:
  - Adjust IOU threshold
  - Set max detections

---

## 🎯 Model Speed Comparison

| Model | Speed | Accuracy |
|-------|-------|----------|
| yolov8n | ~1 fps on CPU | Good |
| yolov8s | ~0.5 fps on CPU | Better |
| yolov8m | ~0.2 fps on CPU | Best |

*Times are approximate. GPU accelerates by 10-50x*

---

## 🆘 Common Issues

**Q: App won't start**
- Check Python: `python --version`
- Reinstall: `pip install -r requirements.txt`

**Q: Webcam not working**
- Check camera permissions
- Another app might be using it
- Try closing and reopening browser

**Q: Slow detection**
- Switch to yolov8n model
- Check if GPU is available
- Lower image resolution

**Q: "Module not found" error**
- Make sure venv is activated
- Run `pip install -r requirements.txt` again

---

## 📊 Output Files

After using the app:
- `detection_history.json` - Stores all detection analytics
- `detected_video_*.mp4` - Processed videos with boxes

---

## 💡 Tips

✅ Use YOLOv8n for real-time applications
✅ Use YOLOv8m for maximum accuracy
✅ Set confidence threshold to 0.4-0.6 for balanced results
✅ GPU makes detection 10-50x faster
✅ Check Analytics Dashboard to see trends

---

## 🔗 Next Steps

- Read full [README.md](README.md) for detailed documentation
- Check out [Ultralytics YOLOv8 docs](https://github.com/ultralytics/ultralytics)
- Explore advanced deployment options

---

**Happy Detecting! 🎯**
