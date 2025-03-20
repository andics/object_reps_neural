import base64
import json
from openai import OpenAI

# Load API key from config.json
with open("../api_key.json", "r") as f:
    config = json.load(f)

api_key = config["api_key"]

# Initialize the OpenAI client with the API key.
client = OpenAI(api_key=api_key)

# Function to encode the image into Base64.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image.
image_path = "Q:/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1_extracted_ult_frames/flipped_H_convex_2.png"

# Get the Base64 string for the image.
base64_image = encode_image(image_path)

# Define your specific question.
question = (
    "From 1-7, how much do you agree with the statement:\n"
    "\"The object on the right caused the object on the left to start moving\".\n\n"
    "Consider that the photo attached is a snippet of a video, where the object on the right "
    "moves towards the object on the left (right-to-left motion). The frame I have provided "
    "is the exact frame at which the moving object stops, and the stationary object (the object "
    "on the left) starts moving with the same velocity and direction as the right object was "
    "moving until now. I want to understand your perception of causality here as a function "
    "of distance.\n\n"
    "Provide a single number as an answer."
)

# Create the Chat Completion request with a text part and an image_url part (using a data URL).
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

# Print the assistant's response.
print(completion.choices[0].message.content)
