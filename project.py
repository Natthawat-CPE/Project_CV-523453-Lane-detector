# นำเข้าโมดูล openCV, numpy, tkinter, filedialog
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

#กำหนดค่าสว่างสำหรับการทำ segmentation
ignore_mask_color = 255

root = tk.Tk()
root.withdraw()

# เลือกเปิดไฟล์ VDO จากเครื่องตามต้องการ 
file_path = filedialog.askopenfilename(title="Choose a video file", filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")])

#เปิดไฟล์ VDO ที่เลือกจากเครื่อง
cap = cv2.VideoCapture(file_path)
# ตั้งค่าเฟรมเรทตามต้องการ
cap.set(cv2.CAP_PROP_FPS, 30)

#เปิด VDO ไปเรื่อยๆจนกว่า VDO จะจบ
while(cap.isOpened()):
    # อ่านค่าเฟรม จาก VDO
    ret, frame = cap.read()
    
    #ตรวจสอบว่าอ่านเฟรมได้สำเร็จหรือไม่
    if ret:
        try:
            # แปลงแต่ละเฟรมให้เป็น GrayScale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # แล้วนำ GrayScale ที่ได้ไปใส่ Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # แล้วนำ Gaussian blur ที่ได้ไปหาขอบด้วย  Canny edge detection
            # โดยมีค่า threshold ที่ยอมรับได้อยู่ระหว่าง [50-150]
            edges = cv2.Canny(blur, 50, 150, apertureSize=3)

            #กำหนดจุดยอดของรูปหลายเหลี่ยม โดยประกอบด้วย tuple 4 ตัว โดยแต่ละตัวเป็นคู่อันดับ x,y ในระบบพิกัดภาพ
            #(x1,y1) : มุมซ้ายล่างของภาพ, (x2,y2) : มุมบนซ้ายของภาพ, (x3,y3) : มุมบนขวาของภาพ, (x4,y4) : มุมขวาล่างของภาพ
            vertices = np.array([[(0, frame.shape[0]), (450, 310), (800, 310), (frame.shape[1], frame.shape[0])]], dtype=np.int32)
            
            #กำหนดให้ตัวแปร mask กำหนดเส้นขอบเขตที่ได้จากการกำหนดจุดยอด
            #ซึ่งจะเป็นภาพแบบ binary mask แต่ละ pixel จะมีค่าเป็น 0,1 เท่านั้น
            #โดยให้ 1 เป็นจุดที่เราจะทำการ พิจารณา
            mask = np.zeros_like(edges)
            
            #สร้างรูปหลายเหลี่ยมบนตัวแปร mask และกำหนดจุดยอดจากตัวแปร vertices 
            #และสีที่ใช้เติมลงไปในรูปหลายเหลี่ยมในที่นี้คือตัวแปร ignore_mask_color = 255 (สีขาว)
            cv2.fillPoly(mask, vertices, ignore_mask_color)
            
            #ทำการเปรียบเทียบ pixel ของ binary image สั้ง 2 (edges, mask)
            #และสร้าง binary image ใหม่มีค่าเป็น 1 เฉพาะ ส่วนที่ edges, mask เป็น 1 เหมือนกัน
            masked_edges = cv2.bitwise_and(edges, mask)

            # นำขอบภาพที่ได้จากการเปรียบเทียบ binary image มาสร้างเส้น lane โดยใช้ Hough transform
            lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

            if lines is not None:
                # จัดกลุ่มเส้นที่ค้นพบจากการใช้ Hough transform เพื่อนำไปใช้สร้างเส้นตรง
                line_groups = {}
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # คำนวณความชัน
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # คำนวณจุดตัดแกน y 
                    y_intercept = y1 - slope * x1
                    
                    if slope < 0:
                        # ถ้าความชัน(m) < 0 ให้เป็นเส้น lane ซ้าย
                        line_groups.setdefault('left', []).append((slope, y_intercept))
                    else:
                        # ถ้าความชัน(m) > 0 ให้เป็นเส้น lane ขวา
                        line_groups.setdefault('right', []).append((slope, y_intercept))

                for group, lines in line_groups.items():
                    # ตรวจสอบว่ามีเส้นตรงในกลุ่มนี้(lines)หรือไม่
                    if len(lines) > 0:
                        #รวม left lane กับ right lane เข้าด้วยกัน
                        slopes, y_intercepts = zip(*lines)
                        # คำนวณค่าเฉลี่ยของ Slope และ y_intercepts จากเส้นตรงกลุ่มนี้
                        avg_slope = np.mean(slopes)
                        avg_y_intercept = np.mean(y_intercepts)
                        # ตรวจสอบว่า slope ไม่เท่ากับ 0 เพื่อหาค่า x1, x2 จากจุดตัดแกน y
                        if avg_slope != 0:
                            x1 = int((frame.shape[0] - avg_y_intercept) / avg_slope)
                            x2 = int((310 - avg_y_intercept) / avg_slope)
                            # วาดเส้นตรงจาก x1, x2 (วาด lane)
                            cv2.line(frame, (x1, frame.shape[0]), (x2, 310), (0, 255, 0), 6)


                    # สร้างพื้นผิวสีเขียวเข้าไปในระหว่างของ left lane และ right lane
                    if 'left' in line_groups and 'right' in line_groups:
                        if len(line_groups['left']) > 0 and len(line_groups['right']) > 0:
                            left_line = line_groups['left'][0]
                            right_line = line_groups['right'][0]
                            pts_left = np.array([[float((frame.shape[0] - left_line[1]) / left_line[0]), frame.shape[0]], 
                                                [float((310 - left_line[1]) / left_line[0]), 310], 
                                                [float((310 - right_line[1]) / right_line[0]), 310], 
                                                [float((frame.shape[0] - right_line[1]) / right_line[0]), frame.shape[0]]], np.int32)
                            green_color = (0, 255, 0) 
                            alpha = 0.1 
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [pts_left], green_color)
                            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


                        # สร้างหัวลูกศรชี้ทางตรงกลางระหว่าง left lane และ right lane
                        if 'left' in line_groups and 'right' in line_groups:
                            if len(line_groups['left']) > 0 and len(line_groups['right']) > 0:
                                left_line = line_groups['left'][0]
                                right_line = line_groups['right'][0]
                                lane_midpoint_x = (left_line[1] - right_line[1]) / (right_line[0] - left_line[0])
                                lane_midpoint_y = left_line[0] * lane_midpoint_x + left_line[1]
                                lane_midpoint = (int(lane_midpoint_x), int(lane_midpoint_y))
                                arrow_tip = (lane_midpoint[0], 400)
                                arrow_tail = (lane_midpoint[0], frame.shape[0])
                                cv2.arrowedLine(frame, arrow_tail, arrow_tip, (0, 0, 255), 3, tipLength=0.2)

            
            
            # แสดงรูปภาพแต่ละเฟรมที่ผ่านกระบวนกสรทั้งหมดที่กล่าวมา
            cv2.imshow('frame2', frame)

            # กำหนด ปุ่มที่ไว้ใช้สำหรับการหยุดทำงานของ program
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # กำหนด exception ที่ส่งผลทำให้โปรแกรมไม่เสถียร
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            print(f"An error occurred: {str(e)}")
            continue
    
    # หากหมดเฟรมเมื่อไหร่ให้หยุดการทำงาน
    else:
        break

# ปล่อยทรัพยากรที่กำลังถูกใช้งาน
cap.release()

# ปิดหน้าต่างของ OpenCV ทั้งหมดที่เปิดอยู่ในขณะนั้น
cv2.destroyAllWindows()
