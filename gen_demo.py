import os
import cv2
import numpy as np
import onnx
import onnxruntime
from functools import lru_cache

# Check if running in Google Colab - (Simplified for local run)
IN_COLAB = False

default_onnx_path = 'centerface.onnx'

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img

class CenterFace:
    def __init__(self, onnx_path=None, in_shape=None, override_execution_provider=None):
        self.in_shape = in_shape
        self.onnx_input_name = 'input.1'
        self.onnx_output_names = ['537','538','539','540']
        if onnx_path is None:
            onnx_path = default_onnx_path
        if not os.path.exists(onnx_path):
             raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        static_model = onnx.load(onnx_path)
        dyn_model = self.dynamicize_shapes(static_model)

        providers = onnxruntime.get_available_providers()
        # Simplified provider selection
        self.sess = onnxruntime.InferenceSession(dyn_model.SerializeToString(), providers=providers)

    @staticmethod
    def dynamicize_shapes(static_model):
        from onnx.tools.update_model_dims import update_inputs_outputs_dims
        inp_dims, out_dims = {}, {}
        for node in static_model.graph.input:
            inp_dims[node.name] = [d.dim_value for d in node.type.tensor_type.shape.dim]
        for node in static_model.graph.output:
            out_dims[node.name] = [d.dim_value for d in node.type.tensor_type.shape.dim]
        inp_dims['input.1'] = ['B',3,'H','W']
        out_dims.update({
            '537': ['B',1,'h','w'],
            '538': ['B',2,'h','w'],
            '539': ['B',2,'h','w'],
            '540': ['B',10,'h','w'],
        })
        return update_inputs_outputs_dims(static_model, inp_dims, out_dims)

    @staticmethod
    @lru_cache(maxsize=8)
    def shape_transform(in_shape, orig_shape):
        h0, w0 = orig_shape
        if in_shape is None:
            w1, h1 = w0, h0
        else:
            w1, h1 = in_shape
        w1 = int(np.ceil(w1/32)*32)
        h1 = int(np.ceil(h1/32)*32)
        sw = w1 / w0
        sh = h1 / h0
        return w1, h1, sw, sh

    def __call__(self, img: np.ndarray, threshold=0.5):
        img = ensure_rgb(img)
        orig_shape = img.shape[:2]
        target_shape_wh = orig_shape[::-1] if self.in_shape is None else self.in_shape
        w1, h1, sw, sh = self.shape_transform(target_shape_wh, orig_shape)
        blob = cv2.dnn.blobFromImage(img, 1.0, (w1,h1), (0,0,0), swapRB=False, crop=False)
        heatmap, scale, offset, lms = self.sess.run(self.onnx_output_names, {self.onnx_input_name: blob})
        dets, decoded_lms = self.decode(heatmap, scale, offset, lms, (h1,w1), threshold)
        if dets.shape[0] > 0:
            dets[:, 0:4:2] = np.clip(dets[:, 0:4:2], 0, w1)
            dets[:, 1:4:2] = np.clip(dets[:, 1:4:2], 0, h1)
            decoded_lms[:, 0:10:2] = np.clip(decoded_lms[:, 0:10:2], 0, w1)
            decoded_lms[:, 1:10:2] = np.clip(decoded_lms[:, 1:10:2], 0, h1)
            dets[:, 0:4:2] /= sw
            dets[:, 1:4:2] /= sh
            decoded_lms[:, 0:10:2] /= sw
            decoded_lms[:, 1:10:2] /= sh
            h0, w0 = orig_shape
            dets[:, 0:4:2] = np.clip(dets[:, 0:4:2], 0, w0)
            dets[:, 1:4:2] = np.clip(dets[:, 1:4:2], 0, h0)
            decoded_lms[:, 0:10:2] = np.clip(decoded_lms[:, 0:10:2], 0, w0)
            decoded_lms[:, 1:10:2] = np.clip(decoded_lms[:, 1:10:2], 0, h0)
        return dets, decoded_lms

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.5):
        heatmap = np.squeeze(heatmap)
        s0_map, s1_map = scale[0,0], scale[0,1]
        o0_map, o1_map = offset[0,0], offset[0,1]
        lm_map = landmark[0]
        stride = 4
        h_out, w_out = heatmap.shape
        size_h, size_w = size
        ys, xs = np.where(heatmap > threshold)
        boxes, lm_list = [], []
        for y, x in zip(ys, xs):
            score = heatmap[y, x]
            cx_feat = x + o1_map[y, x]
            cy_feat = y + o0_map[y, x]
            box_h = np.exp(s0_map[y, x]) * stride
            box_w = np.exp(s1_map[y, x]) * stride
            x1 = max(0, cx_feat * stride - box_w / 2)
            y1 = max(0, cy_feat * stride - box_h / 2)
            x2 = min(x1 + box_w, size_w)
            y2 = min(y1 + box_h, size_h)
            boxes.append([x1, y1, x2, y2, score])
            lm = []
            for j in range(5):
                lm_x = lm_map[j * 2 + 1, y, x] * box_w + x1
                lm_y = lm_map[j * 2,     y, x] * box_h + y1
                lm.append(lm_x)
                lm.append(lm_y)
            lm_list.append(lm)
        if not boxes:
            return np.zeros((0,5),dtype=np.float32), np.zeros((0,10),dtype=np.float32)
        boxes = np.array(boxes, dtype=np.float32)
        lms   = np.array(lm_list, dtype=np.float32)
        keep  = self.nms(boxes[:,:4], boxes[:,4], 0.3)
        return boxes[keep], lms[keep]

    @staticmethod
    def nms(boxes, scores, thresh):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]
        return keep

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def apply_clahe(frame: np.ndarray) -> np.ndarray:
    frame_bgr = ensure_rgb(frame)
    try:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except cv2.error as e:
        return frame_bgr

def apply_blur(roi, blur_mode="auto", gaussian_ksize=25, mosaic_size=10):
    if roi is None or roi.shape[0] == 0 or roi.shape[1] == 0:
        return roi
    h, w = roi.shape[:2]
    area = h * w
    if area == 0: return roi
    if blur_mode == "auto":
        blur_mode = "mosaic" if area > 10000 else "gaussian"
    if blur_mode == "gaussian":
        ksize = gaussian_ksize if gaussian_ksize % 2 != 0 else gaussian_ksize + 1
        ksize = min(ksize, w // 2 * 2 + 1, h // 2 * 2 + 1)
        if ksize < 3: ksize = 3
        if w < ksize or h < ksize : blur_mode = "mosaic"
        else: return cv2.GaussianBlur(roi, (ksize, ksize), 0)
    m_size = max(1, mosaic_size)
    target_w = max(1, w // m_size)
    target_h = max(1, h // m_size)
    small = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def process_and_save(image_path, output_path):
    centerface = CenterFace()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return
    
    processed_frame = img.copy()
    h, w = processed_frame.shape[:2]
    enhanced = apply_clahe(processed_frame)
    dets, lms_raw = centerface(enhanced)
    
    boxes = []
    for x1, y1, x2, y2, score in dets:
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        ww, hh = x2 - x1, y2 - y1
        if ww > 0 and hh > 0:
            boxes.append((x1, y1, ww, hh))
            
    for (x, y, ww, hh) in boxes:
        y_end = min(y + hh, h)
        x_end = min(x + ww, w)
        if y < y_end and x < x_end:
            roi = processed_frame[y:y_end, x:x_end]
            if roi.size > 0:
                blurred_roi = apply_blur(roi, blur_mode="auto")
                processed_frame[y:y_end, x:x_end] = blurred_roi
                
    cv2.imwrite(output_path, processed_frame)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    process_and_save("22_Picnic_Picnic_22_183.jpg", "output_22_Picnic.jpg")
