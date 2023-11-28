import numpy as np

import tracker
from detector import Detector
import cv2

if __name__ == '__main__':

    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    

    # 初始化第一个撞线polygon，上行线
    list_pts_blue = [[833,79],[887,135],[1134,183],[1383,172],[1852,275],
                     [1841,343],[1358,230],[1089,236],[803,179],[765,87]]
                      
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis] 
    # comment by lanny
    # cv2.fillPoly可以就理解成塗色，會把上面list中的點都連起來；左上角是00點，x往右增加，y往下增加。

    # 填充第二个polygon，下行线
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[75,316],[96,490],[527,730],[1024,766],[1770,914],
                       [1750,999],[987,818],[494,798],[46,522],[22,330]]
                    
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 填充第三个polygon，中間區域
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_gray=[[85,322],[118,493],[546,734],[1040,767],[1803,921],
                   [1894,524],[1872,368],[1388,244],[1100,253],[822,193],[778,101]]
    
    ndarray_pts_gray = np.array(list_pts_gray, np.int32)
    polygon_gray_value_3 = cv2.fillPoly(mask_image_temp, [ndarray_pts_gray],color=3)
    polygon_gray_value_3=polygon_gray_value_3[:,:,np.newaxis]

    # 撞线检测用mask，包含3个polygon，（值范围 0、1、2、3），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2+ polygon_gray_value_3

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 灰色圖片
    gray_color_plate = [102, 102, 153]
    gray_image = np.array(polygon_gray_value_3 * gray_color_plate,np.uint8)


    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image + gray_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))
    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []
    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []
    # list 與灰色區域重疊
    list_overlapping_gray_polygon = []

    # 进入数量
    enter_count = 0
    # 离开数量
    leave_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./video/test.mp4')
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                        print(f'All blue:{list_overlapping_blue_polygon}')
                    pass # 如果已经撞过蓝色的线，就什么都不做，表示还没走出蓝色线范围

                    if track_id in list_overlapping_gray_polygon:
                        leave_count += 1
                        print(f'类别: {label} | id: {track_id} | 从上方离开区域 | 离开人数: {leave_count} | 图中人物：{list_overlapping_gray_polygon}')
                        list_overlapping_gray_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass
                    
                    if track_id in list_overlapping_gray_polygon:
                        
                        leave_count += 1
                        print(f'类别: {label} | id: {track_id} | 从下方离开区域 | 离开人数: {leave_count} | 图中人物：{list_overlapping_gray_polygon}')
                        list_overlapping_gray_polygon.remove(track_id)
                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                
                elif polygon_mask_blue_and_yellow[y, x] == 3:

                    if track_id not in list_overlapping_gray_polygon:
                        list_overlapping_gray_polygon.append(track_id)
                    pass
                
                    if track_id in list_overlapping_blue_polygon:
                        enter_count+=1
                        print(f'类别: {label} | id: {track_id} | 从上方进入区域 | 进入人数: {enter_count} | 图中人物：{list_overlapping_gray_polygon}')
                        list_overlapping_blue_polygon.remove(track_id)
                    elif track_id in list_overlapping_yellow_polygon:
                        enter_count+=1
                        print(f'类别: {label} | id: {track_id} | 从下方进入区域 | 进入人数: {enter_count} | 图中人物：{list_overlapping_gray_polygon}')
                        list_overlapping_yellow_polygon.remove(track_id)
                    else:
                        # 表示人物未曾经过蓝色或黄色区域，突然出现在中间灰色区域中
                        # 有两种可能：一种是原本就在的人物，一种是跟踪失败的人物
                        # 先不作处理
                        pass
                            
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            list_overlapping_gray_polygon.clear()
            pass
        pass

        text_draw = 'Enter: ' + str(enter_count) + \
                    ' , Leave: ' + str(leave_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()
