import cv2
import tritonclient.grpc as grpcclient
import numpy as np




class TrafficSignDetector:

    def __init__(self, url: str, detector_model_name: str, classification_model_name: str):
        """
        Parameters
        ----------
        url: triton inference server url
        detector_model_name: detection model name in triton
        classification_model_name: classification model name in triton
        """
        self.client = grpcclient.InferenceServerClient(url)
        self.detector_model_name = detector_model_name
        self.classification_model_name = classification_model_name

    def __call__(self, image):
        prepr_img = self.detection_preprocess(image)
        detection_res = self.call_detection(prepr_img)
        class_preprocess, boxes = self.classification_preprocess(image, detection_res)
        cls_res = self.call_classification(class_preprocess)
        res_img = self.postprocess(image, boxes, cls_res)
        return res_img

    def call_detection(self, image) -> np.ndarray:

        inputs = [
            grpcclient.InferInput(
               'images', image.shape, "FP32"
            )
        ]
        #image_fp32 = image.astype(np.float32)
        outputs = []
        inputs[0].set_data_from_numpy(image)
        detector_res = self.client.infer(
            model_name='detection',
            inputs=inputs,
            outputs=outputs,
        )
        output_array = detector_res.as_numpy('output0')
        return output_array

    def call_classification(self, image):
        inputs = [
            grpcclient.InferInput(
                'images', image[0].shape, "FP32"
            )
        ]
        outputs = []
        inputs[0].set_data_from_numpy(image[0])
        detector_res = self.client.infer(
            model_name='classification',
            inputs=inputs,
            outputs=outputs,
        )
        output_array = detector_res.as_numpy('output0')
        class_names = [str(i) for i in range(188)] # тут можно задать реальные названия - стоп, переход и т.д.
        max_prob_index = np.argmax(output_array)
        predicted_class = class_names[max_prob_index]
        return predicted_class

    def detection_preprocess(self, img):

        expected_width = 1280
        expected_height = 1280
        original_image: np.ndarray = img
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        input_image = cv2.resize(image, (expected_width, expected_height))
        input_image = (input_image / 255.0).astype(np.float32)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def preprocess_batch(self, batch):
        expected_width = 1280
        expected_height = 1280

        preprocessed_batch = []

        for image in batch:
            [height, width, _] = image.shape
            length = max((height, width))
            new_image = np.zeros((length, length, 3), np.uint8)
            new_image[0:height, 0:width] = image
            input_image = cv2.resize(new_image, (expected_width, expected_height))
            input_image = (input_image / 255.0).astype(np.float32)
            input_image = input_image.transpose(2, 0, 1)
            input_image = np.expand_dims(input_image, axis=0)
            preprocessed_batch.append(input_image)

        return preprocessed_batch
    def classification_preprocess(self, img, detection_res):
        outputs = np.array([cv2.transpose(img[0])])
        image_height, image_width, _ = img.shape
        #outputs = (detection_res)
        rows = outputs.shape[1]
        boxes = []
        scores = []

        for i in range(rows):
            x, y, w, h = outputs[0][i][:4]

            score = outputs[0][i][4]
            if score > 0.5:
                left = int((x))
                top = int((y))
                width = int(w)
                height = int(h)
                box = np.array([left, top, width, height])
                boxes.append(box)
                scores.append(score)
        if boxes:
            image_batch = [img[int(y):int(y + h), int(x):int(x + w)] for x, y, w, h in boxes]
            preprocessed_batch = self.preprocess_batch(image_batch)
            return preprocessed_batch, boxes


    def postprocess(self, image, detection_res, class_res):
        boxes = detection_res
        class_names = class_res
        for box, class_name in zip(boxes, class_names):
            print(box)
            x, y, w, h = map(int, box)
            label = f"{class_name}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image


if __name__ == '__main__':
    traffic_sign_recognizer = TrafficSignDetector("10.211.1.2:8001",
                                                  "detection",
                                                  "classification")
    img = cv2.imread("img.png")
    res_img = traffic_sign_recognizer(img)
    cv2.imwrite("res.png", res_img)

