
from torch import manual_seed
from torch.nn.functional import cosine_similarity
from skimage import metrics
import cv2
import numpy
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, PreTrainedModel

def compare(baseline_image: Image.Image, optimized_image: Image.Image):
    clip = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to("cuda")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    manual_seed(0)

    def convert_to_array(image: Image.Image):
        # Convert PIL Image to numpy array
        return numpy.array(image.convert("RGB"))

    def clip_embeddings(image: numpy.ndarray):
        processed_input = processor(images=image, return_tensors="pt").to("cuda")
        return clip(**processed_input).image_embeds.to("cuda")

    # Convert PIL images to numpy arrays
    baseline_array = convert_to_array(baseline_image)
    optimized_array = convert_to_array(optimized_image)

    # Convert to grayscale for structural similarity
    grayscale_baseline = cv2.cvtColor(baseline_array, cv2.COLOR_RGB2GRAY)
    grayscale_optimized = cv2.cvtColor(optimized_array, cv2.COLOR_RGB2GRAY)

    # Calculate structural similarity
    structural_similarity = metrics.structural_similarity(
        grayscale_baseline, 
        grayscale_optimized, 
        full=True
    )[0]

    # Clean up grayscale arrays
    del grayscale_baseline
    del grayscale_optimized

    # Get CLIP embeddings and calculate similarity
    baseline_embeddings = clip_embeddings(baseline_array)
    optimized_embeddings = clip_embeddings(optimized_array)

    clip_similarity = cosine_similarity(baseline_embeddings, optimized_embeddings)[0].item()

    # Return weighted combination of similarities
    return clip_similarity * 0.35 + structural_similarity * 0.65


if __name__=="__main__":
    from PIL import Image
    from math import sqrt

    # Load images
    baseline_img = Image.open("/root/work/flux-fp8-api/output.jpg")
    optimized_img = Image.open("/root/work/flux-schnell-edge-inference/src/src.jpg")

    # Calculate similarity
    f_similarity = compare(baseline_img, optimized_img)

    print(f_similarity)

    scale = 1 / (1 - 0.7)
    similarity = sqrt((f_similarity - 0.7) * scale)

    print(similarity)
