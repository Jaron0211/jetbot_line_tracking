import cv2
import numpy as np
from numpy.linalg import inv, norm
from math import acos, pi
import PID_v2_1 as Pv
import PID_v2_2 as PID
#with np.load('calibration.npz') as file:
    #mtx, dist, rvecs, tvecs = [file[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

from jetbot import Robot


def preprocessing(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # lower hls range [0, 105, 0] higher hls range [255, 155, 255]
    lower = np.array([0, 120, 0])
    higher = np.array([255, 255, 255])
    mask = cv2.inRange(hls_img, lower, higher)
    return mask


def polyfit(idx):
    if len(idx) < 3:
        parameters = []
    else:
        v = idx[:, 0]
        u = idx[:, 1]
        parameters = np.polyfit(v, u, 2)
    return parameters


def detect_lane_line(binary_image):
    left = binary_image[:, 0:200]
    right = binary_image[:, 200:400]
    idx_left = np.transpose(np.nonzero(left))
    idx_right = np.transpose(np.nonzero(right))
    for i in range(0, len(idx_right)):
        idx_right[i, 1] += 200
    left_parameter = polyfit(idx_left)
    right_parameter = polyfit(idx_right)
    return left_parameter, right_parameter


def calculate_lane_line(image, fit):
    if len(fit) != 0:
        h, w, dim = image.shape
        y = np.linspace(0, h - 1, h)
        x = fit[0] * y ** 2 + fit[1] * y + fit[2]
        pts = np.transpose(np.vstack([x, y])).astype(int)
    else:
        pts = np.array([[0, 0], [0, 0]])
    return pts


def detect_center(fit_l, fit_r):
    y_c = 300
    if (len(fit_l) != 0) & (len(fit_r) == 0):
        x_lc = fit_l[0] * y_c + fit_l[1] * y_c + fit_l[2]
        x_c = int(x_lc + 100)
    elif (len(fit_l) == 0) & len(fit_r) != 0:
        x_rc = fit_r[0] * y_c + fit_r[1] + y_c + fit_r[2]
        x_c = int(x_rc - 100)
    elif (len(fit_l) == 0) & (len(fit_r) == 0):
        x_c = 0
    else:
        x_lc = fit_l[0] * y_c + fit_l[1] * y_c + fit_l[2]
        x_rc = fit_r[0] * y_c + fit_r[1] + y_c + fit_r[2]
        x_c = int((x_lc + x_rc) / 2)
    return x_c, y_c


def plot_lane_line(frame, pts1, pts2, Minv):
    h, w, dim = frame.shape
    mask = np.zeros((400, 400, 3), np.uint8)
    mask = cv2.polylines(mask, pts=[pts1], isClosed=False, color=(0, 0, 255), thickness=3)
    mask = cv2.polylines(mask, [pts2], False, (0, 0, 255), 3)
    inv_perspective = cv2.warpPerspective(mask, Minv, (w, h))
    image = cv2.addWeighted(frame, 1, inv_perspective, 0.8, 1)
    return image


def b2im(x, y, z, mtx):
    w = np.array([[x-5.74], [y+2.83], [z], [1]])
    R = np.array([[0.04306311, -0.99895377, -0.01539257],
                  [-0.52110037, -0.00931324, -0.85344459],
                  [0.85240833, 0.04477305, -0.52095624]])
    T = np.array([[3.7615314],
                  [1.8150472],
                  [8.94055666]])
    RT = np.hstack([R, T])
    im = mtx.dot(RT.dot(w))
    u_b = int(im[0] / im[2])
    v_b = int(im[1] / im[2])
    s = im[2]
    return u_b, v_b, s


def detect_angle(xc, yc, img):
    c = np.array([xc, yc])
    C = (xc, yc)
    o = np.array([415, 859])
    A = (o[0], o[1])
    b = np.array([382, 320])
    B = (b[0], b[1])
    #
    e1 = norm(c-o)
    e2 = norm(b-o)
    angle = acos(e2/e1)
    angle = angle*180/pi
    #
    cv2.line(img, A, C, (0, 255, 255), 3) # ¤è¦V½u
    img = cv2.line(img, A, B, (0, 255, 0), 3) # ¤¤½u
    return img, angle


def gstreamer_pipeline(
        w=640,
        h=480,
        display_width=640,
        display_height=480,
        framerate=60,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                w,
                h,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def show_camera():
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        # undistortion matrix
        #newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 1, (640, 480))
        roi_lane = np.array([[0, 480], [640, 480], [540, 200], [150, 200]])
        dst_pts = np.float32([[0, 400], [400, 400], [400, 0], [0, 0]])
        M = cv2.getPerspectiveTransform(np.float32(roi_lane), dst_pts)
        Minv = inv(M)
        i = 0
        while 1:
            ret, frame = cap.read()
            perspective_image = cv2.warpPerspective(frame, M, (400, 400))
            thresh = preprocessing(perspective_image)
            left_fit, right_fit = detect_lane_line(thresh)
            pts_left = calculate_lane_line(frame, left_fit)
            pts_right = calculate_lane_line(frame, right_fit)
            line = plot_lane_line(frame, pts_left, pts_right, Minv)
            xc, yc = detect_center(left_fit, right_fit)
            cv2.circle(line, (xc, yc), 3, (0, 255, 0), -1)
            line, angle = detect_angle(xc, yc, line)
            print(angle)
            cv2.imshow('line', line)
            cv2.imshow('binary', thresh)

            # setup PID
            attitude_ctrl = PID.pid(0.5,0,0.1)
            # obtain state
            #P_end_i = np.array([xc,yc])
            #P_end_w = Pv.imgpt2wrd(P_end_i)
            #P_loc_i = np.array([415, 859])
            #P_loc_w = Pv.imgpt2wrd(P_loc_i)
            #theta, y = Pv.sensor(P_loc_w, P_end_w)

            # attitude control
            attitude_ctrl.cur = angle
            # attitude_ctrl.desire = 0
            attitude_ctrl.cal_err()
            r_cmd = attitude_ctrl.output()
            RPSR, RPSL = Pv.differential_cal(r_cmd)
            Pv.motor_ctrl(RPSR, RPSL)

            # =============================================================
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    show_camera()