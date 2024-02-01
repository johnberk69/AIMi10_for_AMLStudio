# 01/31/2024 JohnB
# example call >python predict.py TCAI_AIM_Iteration_10_on_Gen_Compact_S1.ONNX/model.onnx 487a.jpg<Enter>

# example results
# Label: ElkCowNight, Probability: 0.06088, box: (0.74602, 0.60209) (0.79751, 0.73416)
# Label: ElkBullNight, Probability: 0.02869, box: (0.05709, 0.13092) (0.94145, 0.82532)
# Label: DeerDoeNight, Probability: 0.01669, box: (0.83914, 0.41981) (1.01059, 0.70849)
# Label: ElkBullNight, Probability: 0.01530, box: (0.07913, -0.01355) (0.98032, 0.48434)
# Label: WolfNight, Probability: 0.01407, box: (-0.02169, 0.77771) (0.72656, 1.01300)
# Label: BearNight, Probability: 0.01319, box: (0.21241, -0.03850) (0.79951, 1.02480)
# Label: SwineNight, Probability: 0.01228, box: (0.01871, 0.42287) (0.10995, 0.53394)
# Label: DeerDoeNight, Probability: 0.01215, box: (0.00528, 0.38693) (0.52865, 1.07206)
# Label: MooseBullNight, Probability: 0.01105, box: (0.44823, 0.45687) (0.59371, 0.66333)
# Label: BearNight, Probability: 0.01003, box: (0.09423, 0.73879) (0.15411, 0.78231)

# IMPORTS
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image

PROB_THRESHOLD = 0.02  # Minimum probably to show results.


class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

# these classes need to be edited to match the model's associated labels.txt file.  NOTE, label file's line numbers are NOT these Index numbers
def switch(class_id):
    if class_id == 0:
        return "BearDay"
    elif class_id == 1:
        return "BearNight"
    elif class_id == 2:
        return "CougarDay"
    elif class_id == 3:
        return "CougarNight"
    elif class_id == 4:
        return "CoyoteDay"
    elif class_id == 5:
        return "CoyoteNight"
    elif class_id == 6:
        return "CraneDay"
    elif class_id == 7:
        return "DeerBuckDay"
    elif class_id == 8:
        return "DeerBuckNight"
    elif class_id == 9:
        return "DeerDoeDay"
    elif class_id == 10:
        return "DeerDoeNight"
    elif class_id == 11:
        return "ElkBullDay"
    elif class_id == 12:
        return "ElkBullNight"
    elif class_id == 13:
        return "ElkCowDay"
    elif class_id == 14:
        return "ElkCowNight"
    elif class_id == 15:
        return "HumanDay"
    elif class_id == 16:
        return "HumanNight"
    elif class_id == 17:
        return "MooseBullDay"
    elif class_id == 18:
        return "MooseBullNight"
    elif class_id == 19:
        return "MooseCowDay"
    elif class_id == 20:
        return "MooseCowNight"
    elif class_id == 21:
        return "SwineDay"
    elif class_id == 22:
        return "SwineNight"
    elif class_id == 23:
        return "TurkeyDay"
    elif class_id == 24:
        return "TurkeyNight"
    elif class_id == 25:
        return "VehicleDay"
    elif class_id == 26:
        return "VehicleNight"
    elif class_id == 27:
        return "WolfDay"
    elif class_id == 28:
        return "WolfNight"


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            clabel = switch(class_id)
            print(f"Label: {clabel}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('image_filepath', type=pathlib.Path)

    args = parser.parse_args()

    model = Model(args.model_filepath)
    outputs = model.predict(args.image_filepath)
    print_outputs(outputs)


if __name__ == '__main__':
    main()
    
    