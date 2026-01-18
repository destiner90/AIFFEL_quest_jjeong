def sticker_attach(img_path, sticker_img_path):
    # 필요한 패키지 import 하기
    import os 
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import dlib
    
    home_dir = os.getenv('HOME')
    my_image_path = img_path
    img_bgr = cv2.imread(my_image_path)
    img_show = img_bgr.copy()
    # plt.imshow(img_bgr)
    # plt.show()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector_hog = dlib.get_frontal_face_detector()
    dlib_rects = detector_hog(img_rgb, 1)

    for dlib_rect in dlib_rects:
        l = dlib_rect.left()
        t = dlib_rect.top()
        r = dlib_rect.right()
        d = dlib_rect.bottom()

        cv2.rectangle(img_show, (l, t), (r, d), (0, 255, 0), 2, lineType=cv2.LINE_AA)

    img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

    model_path = os.path.join(home_dir, 'work/camera_sticker/models/shape_predictor_68_face_landmarks.dat')
    landmark_predictor = dlib.shape_predictor(model_path)

    list_landmarks = []

    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))

        list_landmarks.append(list_points)

    # 여러 개의 얼굴이 검출됐다는 가정하에 list_landmarks는 리스트 형태임
    for landmark in list_landmarks:
        for point in landmark:
            cv2.circle(img_show, point, 2, (0, 255, 255), -1)

    img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_show_rgb)
    # plt.show()

    
    # Step 3. 스티커 적용 위치 확인하기
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        # print(landmark[33]) # 코의 인덱스는 30
        x = landmark[33][0]
        y = landmark[33][1] + dlib_rect.height() // 2
        w = h = dlib_rect.width()
    
        # print(f"(x,y) : ({x}, {y})")
        # print(f"(w,h) : ({w}, {h})")


    # Step 4. 스티커 적용하기
    sticker_path = sticker_img_path
    img_sticker = cv2.imread(sticker_path)
    img_sticker = cv2.resize(img_sticker, (w, h))
    # print(img_sticker.shape)

    refined_x = x - w // 2
    refined_y = y - h

    # 이미지가 밖으로 시작하는 경우 예외처리
    if refined_x < 0:
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0
        
    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:, :]
        refined_y = 0

    # 오른쪽, 아래쪽으로 이미지가 넘어가는 경우 예외처리
    
    H, W = img_show.shape[:2]
    sh, sw = img_sticker.shape[:2]
    
    # 스티커가 놓일 끝 좌표
    end_x = refined_x + sw
    end_y = refined_y + sh
    
    # 오른쪽/아래쪽이 넘어가면 스티커를 잘라낸다
    if end_x > W:
        img_sticker = img_sticker[:, :W - refined_x]
    if end_y > H:
        img_sticker = img_sticker[:H - refined_y, :]


    sticker_area = img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

    img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x + img_sticker.shape[1]] = np.where(img_sticker==255, sticker_area, img_sticker).astype(np.uint8)

    
    sticker_area = img_bgr[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
    img_bgr[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==255, sticker_area, img_sticker).astype(np.uint8)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return plt.show()