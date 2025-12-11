# 🏋️ AI Fitness Counter: Real-Time Pose Estimation Trainer

## 🌟 Project Overview

This project uses the YOLOv8 pose estimation model with OpenCV and NumPy to create a real-time, AI-powered fitness counter. It tracks a user's key joint angles (knees, elbows, hips) via a webcam to automatically count repetitions for specific exercises and provide instantaneous form correction feedback.

### Key Features

* **Dual Exercise Tracking:** Supports rep counting and stage tracking for **Push-ups** and **Squats**.
* **Real-Time Feedback:** Displays current rep count, exercise stage (`UP`/`DOWN`), and angle measurement.
* **Advanced Form Correction (Push-ups):** Checks the body line angle (Shoulder-Hip-Knee) to ensure a straight back. Reps are only counted if the form is correct.
* **Side Robustness:** Automatically selects the most visible side of the body (Left or Right) for accurate angle calculation.
* **Interactive Controls:** Allows switching between exercises using keyboard shortcuts.

---

## 💻 Requirements

To run this project, you need Python and the following libraries:

### Python Libraries

Install the required packages using pip:

``bash
pip install ultralytics opencv-python numpy

Model Weights 

This project requires a pre-trained YOLOv8 pose estimation model

Download the Weights: Ensure you have the model weights (e.g., best.pt or a standard YOLOv8 pose model).

Model Path: The current code expects the model file at the path used during development:./runs/pose/fitness_tracker_yolov8/weights/best.ptNote: Adjust the path in squat_counter.py 
if your model is located elsewhere.

🚀 How to Run the Project 

Ensure a Webcam is Connected: The application uses cv2.VideoCapture(0) for the default camera.

Save the Code: Save the Python code (we developed) as a file named ai_fitness_counter.py (or similar).Execute the Script:Bashpython ai_fitness_counter.py

Maximize the Window: The application is configured to run in fullscreen mode for better viewing.

⌨️ Controls Use these keys while the application window is active: 

Key 'p' Action Switch to Push-up counting mode.
's' Switch to Squat counting mode.
'q' Quit the application and close the window.

📐 How the Counting Logic Works: 
The system calculates the relevant joint angle in real-time and uses state-based thresholds to count reps.

1. Push-ups Joints Tracked: Shoulder, Elbow, Wrist (measures elbow bend)
2. 
3. Counting Logic:UP Stage (Reset): Elbow angle must be 165.0 (arm straight).
   
5. DOWN Stage (Rep Count): Elbow angle must be $< 135.0 (bent arm), and the stage must be UP.
 
7. Form Correction: Checks the Hip angle (Shoulder-Hip-Knee).
   
9. If the angle is 160.0 (indicating sagging or piking), the rep is not counted, and "Correction: Keep your body straight!"
    
11. feedback is given.2. SquatsJoints Tracked: Hip, Knee, Ankle (measures knee bend).
    
13. Counting Logic: UP Stage (Reset): Knee angle must be $> 170.0^ (leg straight/locked out).
    
15. DOWN Stage (Rep Count): Knee angle must be $< 100.0^\circ$ (deep squat), and the stage must be UP.
 
🔮 Future EnhancementsSquat Form Correction:

Implement checks for back straightness (Torso Angle) and knee over toe tracking. 

Exercise Selection Menu: Use a simple UI instead of key presses to select exercises. 

Calibration: Allow users to set their own thresholds based on individual flexibility and body type.

Logging: Save workout data (reps, time, form errors) to a CSV file.
