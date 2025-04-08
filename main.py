import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_image(image, tile_size, overlap):
    """ 이미지를 겹치는 타일로 분할 """
    h, w, _ = image.shape
    step = tile_size - overlap  # 겹침을 고려한 이동 거리
    tiles = []
    tile_positions = []
    
    for y in range(0, h - overlap, step):
        for x in range(0, w - overlap, step):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            tile_positions.append((x, y))
    
    return tiles, tile_positions

def run_yolo_on_tiles(tiles, net, output_layers, classes, tile_positions, conf_threshold=0.5, nms_threshold=0.2): 
    ## conf_th : 객체로 인식될 최소 확률 설정 / 값이 높을 수록 더 정확한 객체 감지.
    ## nms : 중복된 바운딩박스 제거 / 값이 낮으면 중복 감지된 박스가 더 많이 제거.


    """ 각 타일에서 YOLO 객체 감지 수행 후 원본 이미지 좌표로 변환 """
    boxes, confidences, class_ids = [], [], []
    
    for i, tile in enumerate(tiles):
        h, w, _ = tile.shape
        x_offset, y_offset = tile_positions[i]
        
        blob = cv2.dnn.blobFromImage(tile, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * w) + x_offset
                    center_y = int(detection[1] * h) + y_offset
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    x = int(center_x - bw / 2)
                    y = int(center_y - bh / 2)
                    
                    boxes.append([x, y, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def iou(box1, box2):
    """ 두 바운딩 박스 간 IoU 계산 """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def remove_duplicate_boxes(boxes, confidences, class_ids, iou_threshold=0.5):   ## 중복된 바운딩박스 제거 기준
    """ 타일 경계 중복 제거 """                                                  ## 값이 낮으면 더 많은 중복 박스 제거
    new_boxes, new_confidences, new_class_ids = [], [], []
    
    for i in range(len(boxes)):
        is_duplicate = False
        for j in range(len(new_boxes)):
            if iou(boxes[i], new_boxes[j]) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            new_boxes.append(boxes[i])
            new_confidences.append(confidences[i])
            new_class_ids.append(class_ids[i])
    
    return new_boxes, new_confidences, new_class_ids

def draw_boxes(image, boxes, confidences, class_ids, classes, nms_threshold=0.3):
    """ 중복 감지를 제거하고 바운딩 박스를 그림 """
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, nms_threshold)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

# YOLO 설정
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# ✅ GPU 가속 활성화
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# COCO 클래스 로드
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# 고해상도 이미지 로드
image = cv2.imread("photostudio_2k.jpg")
if image is None:
    print("이미지를 로드할 수 없습니다.")
    exit()

# ✅ 타일 크기 자동 조정
n = 4  # 타일 개수 조정 변수
tile_size = min(image.shape[:2]) // n  # 이미지 크기에 따라 동적으로 설정
overlap = tile_size // 5  # 오버랩 비율 설정

# 이미지 타일링
tiles, tile_positions = split_image(image, tile_size, overlap)

# YOLO 실행
boxes, confidences, class_ids = run_yolo_on_tiles(tiles, net, output_layers, classes, tile_positions)

# ✅ 타일 간 중복 제거
boxes, confidences, class_ids = remove_duplicate_boxes(boxes, confidences, class_ids)

# 바운딩 박스 출력
result_image = draw_boxes(image, boxes, confidences, class_ids, classes)

# 결과 이미지 저장
output_filename = "detected_result5.jpg"
cv2.imwrite(output_filename, result_image)
print(f"결과 이미지 저장 완료: {output_filename}")

# 결과 출력
image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
