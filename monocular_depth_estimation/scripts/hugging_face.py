from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os 
from functools import wraps
import time

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f seconds' % \
          (f.__name__, te-ts))
        return result
    return wrap

def read_images(dir, suffix):
    
    image_paths = []
    image_names = []

    for file in os.listdir(dir):
        if file.endswith(suffix):
            image_paths.append(os.path.join(dir, file))
            image_names.append(file)
    
    images = [
        Image.open(image_path).convert("RGB")
        for image_path in image_paths
    ]

    return images, image_names

def load_models(model_name, cache_dir="/repo/checkpoints/huggingface", device="cpu"):
    """
    Load the feature extractor and the model from the Hugging Face Hub.
    """

    feature_extractor = GLPNFeatureExtractor.from_pretrained(
        model_name, cache_dir=cache_dir)
    
    model = GLPNForDepthEstimation.from_pretrained(
        model_name, cache_dir=cache_dir).to(device)
    
    return feature_extractor, model

@timeit
def predict_depth(image, feature_extractor, model):
    """
    Predict the depth of an image.
    """

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    return depth

def concatenated_rgb_and_depth(image, depth):
    """
    Concatenate the RGB image and the depth image side by side.
    """

    # Create a new blank image
    concatenated_image = Image.new('RGB', (image.width + depth.width, max(image.height, depth.height)))

    # Paste RGB image onto the new image
    concatenated_image.paste(image, (0, 0))

    # Paste depth image onto the new image
    concatenated_image.paste(depth, (image.width, 0))

    return concatenated_image
 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "vinvino02/glpn-nyu"
    output_dir = "./output/glpn-nyu"
    data_dir = "/datasets/mono_depth_estimation"

    images, image_names = read_images(data_dir, ".png")

    feature_extractor, model = load_models(model_name, device=device)

    os.makedirs("./output/glpn-nyu", exist_ok=True)
    for image, image_name in zip(images, image_names):
        depth = predict_depth(image, feature_extractor, model)
        # depth.save(os.path.join(output_dir, image_name))
        concatenated_image = concatenated_rgb_and_depth(image, depth)
        concatenated_image.save(os.path.join(output_dir, image_name))

