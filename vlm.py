import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import yaml

# Load and instanciate model
model_path = "/datas01/t0184411/checkpoints/granite-vision-3.2-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
processor = AutoProcessor.from_pretrained(model_path)

# Define prompt
prompt = "Describe the image in one sentence maximum"

# Load images
image_path = "/home/ilan/Hackathon/Drone-Defense-Hackathon/test/17639990872259045.jpg"
image = Image.open(image_path).convert('RGB')

# Load the YAML file & Extract the prompt text
with open("prompts.yaml", "r") as f:
    data = yaml.safe_load(f)
prompt_template = data["prompt_vlm"]


def put_into_template(processor, prompt_text, pil_image):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    return inputs

def infer(model, processor, prompt_template, drone_metadata, pil_image, device, max_new_tokens=1000):
    prompt = (prompt_template.replace("<<drone_metadata>>", drone_metadata))
    inputs = put_into_template(processor, prompt, pil_image).to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(output[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    print("\n\n", response)
    return response


drone_metadata = """ 
    "latitude": 45.4215,
    "longitude": -75.6972,
    "altitude": 150.5,
    "timestamp": "2025-11-24T14:30:00Z",
    """
infer(model, processor, prompt_template, drone_metadata, image, model.device)