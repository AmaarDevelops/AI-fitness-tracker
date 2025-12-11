from ultralytics import YOLO
import numpy as np
import cv2


model = YOLO('./runs/pose/fitness_tracker_yolov8/weights/best.pt')

# -------- Squat joints ---------

# Left side
L_HIP = 11
L_KNEE = 13
L_ANKLE = 15

# Right side
R_HIP = 12
R_KNEE = 14
R_ANKLE = 16

# ----------- Push up joints --------

L_SHOULDER = 5
L_ELBOW = 7
L_WRIST = 9

# Right
R_SHOULDER = 6
R_ELBOW = 8
R_WRIST = 10


# ---------- Mathematical calculations -------
def calculate_3pt_angle(keypoints,p1_index,p2_index,p3_index):
    try:
        # p1 : hip , p2 : Knee , p3 : Ankle
        p1 = keypoints[p1_index]
        p2 = keypoints[p2_index]
        p3 = keypoints[p3_index]

        # Vectors

        vector_a = p1 - p2 # Knee to hip
        vector_b = p3 - p2 # Knee to ankle

        # Calculate the dot product (A . B)
        dot_product = np.dot(vector_a,vector_b)

        # Calculate the magnitude (|A| and |B|)
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        # Calculate the cosine and angle in degrees
        cosine_angle = np.clip(dot_product / (magnitude_a * magnitude_b) , -1.0,1.0)

        # Convert to degrees
        angle_rad = np.arccos(cosine_angle)

        angle_deg = np.degrees(angle_rad)

        return angle_deg

    except Exception as e:
        print(f'Error occured in calculation :- {e}')


# ------------- Side robustness -----------------

def select_best_side_for_exercise(results,p1_idx_L,p2_idx_L,p3_idx_L,p1_idx_R,p2_idx_R,p3_idx_R):
    # Only consider the first detected person
    if len(results[0].keypoints.xy) == 0:
        return 0,0,0,None # No person detected

    # If detects a person, convert into a numpy array
    keypoints_conf = results[0].keypoints.conf[0].cpu().numpy()

    min_conf = 0.5

    def get_size_conf(idx1,idx2,idx3):
        if (keypoints_conf[idx1] < min_conf or keypoints_conf[idx2] < min_conf or
            keypoints_conf[idx3] < min_conf):
            return 0.0

        return (keypoints_conf[idx1] + keypoints_conf[idx2] + keypoints_conf[idx3])

    # Calculate confidence for both sides
    left_conf = get_size_conf(p1_idx_L,p2_idx_L,p3_idx_L)
    right_conf = get_size_conf(p1_idx_R,p2_idx_R,p3_idx_R)

    keypoints_xy  = results[0].keypoints.xy[0].cpu().numpy()

    if right_conf > left_conf:
        # If right side is more visible / clear
        return p1_idx_R,p2_idx_R,p3_idx_R,keypoints_xy
    elif left_conf > 0.0:
        # Left side is more visible
        return p1_idx_L,p2_idx_L,p3_idx_L,keypoints_xy
    else:
        # If neither sides are clear, left as a fallback
        return p1_idx_L,p2_idx_L,p3_idx_L,keypoints_xy


def main():
    cap = cv2.VideoCapture(0)

    # Rep count variables
    rep_count = 0
    squat_stage = 'up'
    UP_THRESHOLD = 165.0 # angle when leg is straight
    DOWN_THRESOLD = 100.0 # angle where leg is squatted

    feedback_text = 'Get ready!'

    predict_settings = {'imgsz' : 640 , 'verbose' : False}

    # --- Push up variables ---
    pushup_count = 0
    pushup_stage = 'up' # default stage

    # Thresholds for elbow angle
    PUSHUP_THRESHOLD = 165.0 # angle when arm straight
    PUSHDOWN_THRESHOLD = 125.0 # Angle when arm is bent

    # Rep counter alternating
    current_excercise = 'pushup'

    display_count = 0
    display_stage = 'INIT'

    while cap.isOpened():
        success,frame = cap.read()

        if not success:
            break

        results = list(model.predict(frame,**predict_settings))

        # Default display text when no person is detected
        angle_text = ""


        if len(results[0].keypoints.xy) > 0:
                keypoints_xy  = results[0].keypoints.xy[0].cpu().numpy()

                if current_excercise == 'pushup':

                    p1,p2,p3,keypoints_xy  = select_best_side_for_exercise(
                        results,L_SHOULDER,L_ELBOW,L_WRIST,
                        R_SHOULDER,R_ELBOW,R_WRIST
                    )
                    angle_to_check = calculate_3pt_angle(keypoints_xy,p1,p2,p3)

                    up_th = PUSHUP_THRESHOLD
                    down_th = PUSHDOWN_THRESHOLD
                    count_var = pushup_count
                    stage_var = pushup_stage

                    # Form correction
                    if p1 == L_SHOULDER:
                        hip_idx,knee_idx = L_HIP,L_KNEE
                    else:
                        hip_idx,knee_idx = R_HIP,R_KNEE

                    body_angle = calculate_3pt_angle(keypoints_xy,p1,hip_idx,knee_idx) # P1=Shoulder, P2=Hip, P3=Knee

                    if body_angle < 160.0:
                        feedback_text = 'Correction: Keep your body straight'
                        can_count = False
                    else:
                        can_count = True

                elif current_excercise == 'squat':
                    p1,p2,p3,keypoints_xy = select_best_side_for_exercise(
                        results,
                        L_HIP,L_KNEE,L_ANKLE,
                        R_HIP,R_KNEE,R_ANKLE
                    )
                    angle_to_check = calculate_3pt_angle(keypoints_xy,p1,p2,p3)
                    up_th = UP_THRESHOLD
                    down_th = DOWN_THRESOLD
                    count_var = rep_count
                    stage_var = squat_stage

                    can_count = True

                # ------ Generic rep counting stage machine ----

                if keypoints_xy is not None and angle_to_check != 0.0 and can_count:
                  angle_text = f'| Angle: {angle_to_check:.1f} deg'

                  if angle_to_check > up_th:
                    stage_var = 'up'
                    feedback_text = 'Arms straight. Ready'

                  elif angle_to_check < down_th and stage_var == 'up':
                    stage_var = 'down'
                    count_var += 1
                    feedback_text = 'rep counted! Push up!'

                  # --- 3. Update the correct counter / stage
                  if current_excercise == 'pushup':
                      pushup_count = count_var
                      pushup_stage = stage_var

                      display_count = pushup_count
                      display_stage = pushup_stage
                      
                  elif current_excercise == 'squat':
                      rep_count = count_var
                      squat_stage = stage_var

                      display_count = rep_count
                      display_stage = squat_stage
        else:
            feedback_text = "No person found!"

            display_count = pushup_count if current_excercise == 'pushup' else rep_count
            display_stage = pushup_stage if current_excercise == 'pushup' else squat_stage

        # Final display

        final_text = f'{current_excercise.upper()} REPS : {display_count} , Stage : {display_stage.upper()} {angle_text} | {feedback_text}'


        # Simple color for contrast
        cv2.putText(frame,final_text,(50,50),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow('AI square Counter',frame)



        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            current_excercise = 'squat'
            feedback_text = 'Switched to SQUAT mode. Get in position'
        elif key == ord('p'):
            current_excercise = 'pushup'
            feedback_text = 'Switched to Push ups. Get in position'

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()














