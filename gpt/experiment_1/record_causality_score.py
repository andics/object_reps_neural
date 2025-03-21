import base64
import json
import os
import csv
from openai import OpenAI

class MultiModalCSVRequester:
    def __init__(self, csv_file, row_name, image_paths, question):
        """
        Initialize with:
          - csv_file: Path to the CSV file (will be appended to)
          - row_name: A label to be added in the CSV row (e.g. an identifier)
          - image_paths: List of local image file paths to include in the request
          - question: The text prompt/question to send along with the images
        """
        # Set working directory to the script's directory.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        self.csv_file = csv_file
        self.row_name = row_name
        self.image_paths = image_paths
        self.question = question

        # Load API key from ../api_key.json.
        config_path = os.path.join("..", "api_key.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        api_key = config["api_key"]

        # Initialize the OpenAI client.
        self.client = OpenAI(api_key=api_key)

    def encode_image_from_file(self, image_path):
        """Reads and Base64-encodes the image from a local file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def run_request(self):
        """Encodes the images, makes the API call, and appends the result to the CSV."""
        # Encode all images.
        encoded_images = [self.encode_image_from_file(path) for path in self.image_paths]

        # Construct the content list with the text prompt and each image as a data URL.
        content_list = [
            {"type": "text", "text": self.question}
        ]
        for encoded in encoded_images:
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                }
            })

        messages = [{
            "role": "user",
            "content": content_list
        }]

        # Call the Chat Completions API using model "gpt-4o-mini".
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
        )

        # Extract the assistant's response.
        response_text = response.choices[0].message.content

        # Append the result to the CSV file.
        with open(self.csv_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.row_name, response_text])

        print("Response appended to CSV:", response_text)
        return response_text

# Example usage:
if __name__ == "__main__":
    # Specify the CSV file to update, a row name, image paths, and the question text.
    csv_file = "responses.csv"
    row_name = "TestRow1"
    image_paths = [
        "Q:/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1_extracted_ult_frames/flipped_H_convex_2.png",
        "Q:/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1_extracted_ult_frames/flipped_H_convex_2.png",
        "Q:/Projects/Object_reps_neural/Programming/gpt/experiment_1/Exp1_extracted_ult_frames/flipped_H_convex_2.png",
    ]
    question = (
        "What are in these images? Is there any difference between them?"
    )

    requester = MultiModalCSVRequester(csv_file, row_name, image_paths, question)
    requester.run_request()
