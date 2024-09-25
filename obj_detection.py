import cv2
import numpy as np
import time
from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client['fraud_detection']
transactions = db['transactions']

def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3-custom.cfg")  # Use custom cfg
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    if isinstance(out_layers, np.ndarray):
        out_layers = out_layers.flatten().tolist()
    output_layers = [layer_names[i - 1] for i in out_layers]
    return net, output_layers

def detect_objects(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.25:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, confidences, class_ids

def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    cash_counter = 0
    invoice_counter = 0
    fraud_counter = 0
    start_time = time.time()

    # Load YOLO model
    net, output_layers = load_yolo_model()

    # Load class labels
    with open("obj.names", "r") as f:  
        classes = [line.strip() for line in f.readlines()]

    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        indexes, boxes, confidences, class_ids = detect_objects(frame, net, output_layers)

        # Draw bounding boxes and labels
        for i in indexes.flatten():
            box = boxes[i]
            x, y, w, h = box
            class_id = class_ids[i]
            label = classes[class_id]  # Get class name
            confidence = confidences[i]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label and confidence
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Logic for cash and invoice detection
        cash_detected = False
        invoice_issued = False
        
        # Update cash and invoice detection based on detected objects
        for i in indexes.flatten():
            class_id = class_ids[i]
            
            if class_id == 0:  # Assuming class_id 0 corresponds to cash 
                cash_detected = True
            if class_id == 1:  # Assuming class_id 1 corresponds to invoice
                invoice_issued = True

        if cash_detected:
            cash_counter += 1
        if invoice_issued:
            invoice_counter += 1
        else:
            if time.time() - start_time > 15:
                fraud_counter += 1

        # Insert transaction record into MongoDB
        transaction_record = {
            "timestamp": time.time(),
            "cash_transaction": cash_detected,
            "invoice_issued": invoice_issued,
            "fraud": not invoice_issued and (time.time() - start_time > 15)
        }
        transactions.insert_one(transaction_record)

        # Overlay counters on the video
        cv2.putText(frame, f'Cash Transactions: {cash_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Invoices: {invoice_counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Fraud: {fraud_counter}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process videos
process_video('../videos/cash_no_invoice_30sec.mp4', '../outputs/cash1.mp4')
process_video('../videos/multiple_cash_transaction_raw_example.mp4', '../outputs/multiple1.mp4')
