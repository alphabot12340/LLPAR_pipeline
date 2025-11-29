from paddlex import create_model
import os
import time
import re

# Load your pedestrian attribute model
model = create_model(
    model_name="PP-LCNet_x1_0_pedestrian_attribute",
    model_dir="PP-LCNet_x1_0_pedestrian_attribute_infer"
)

image_dir = "./crop_result"
output_dir = "./output"

os.makedirs(output_dir, exist_ok=True)

THRESHOLD = 0.5

# Output file
total_txt_path = os.path.join(output_dir, "total_result.txt")

total_runtime = 0.0
image_count = 0

# Start output file
with open(total_txt_path, "w", encoding="utf-8") as total_f:
    total_f.write("PEDESTRIAN ATTRIBUTE INFERENCE RESULTS\n")
    total_f.write("======================================\n\n")


def clean_label(label):
    """
    Remove Chinese text inside parentheses:  Something(中文) → Something
    """
    return re.sub(r"\([^)]*\)", "", label).strip()


def extract_labels_scores(data):
    """Extract label_names and scores for JSON format."""
    if "res" in data:
        block = data["res"]
        if "label_names" in block and "scores" in block:
            return block["label_names"], block["scores"]
    return None, None


# Process images
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    image_count += 1
    start_time = time.time()

    image_output = []

    for res in model.predict(img_path):

        res.print()
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

        data = res.json
        label_names, scores = extract_labels_scores(data)

        if label_names is None:
            image_output.append("ERROR: Could not extract attributes.\n")
            continue

        # Clean labels (remove Chinese parts)
        clean_names = [clean_label(lbl) for lbl in label_names]

        # Build dict and sort
        attrs = dict(zip(clean_names, scores))
        sorted_attrs = sorted(attrs.items(), key=lambda x: x[1], reverse=True)

        # Over-threshold
        image_output.append(f"Predicted attributes (over {THRESHOLD:.2f}):")
        for label, score in sorted_attrs:
            if score >= THRESHOLD:
                image_output.append(f"  {label}: {score:.4f}")

        # All attributes
        image_output.append("\nAll attribute probabilities:")
        for label, score in sorted_attrs:
            image_output.append(f"  {label}: {score:.4f}")

    end_time = time.time()
    elapsed = end_time - start_time
    total_runtime += elapsed

    # Write to total_result.txt
    with open(total_txt_path, "a", encoding="utf-8") as total_f:
        total_f.write(f"Image: {filename}\n")
        total_f.write(f"Processing time: {elapsed:.4f} seconds\n\n")
        total_f.write("\n".join(image_output))
        total_f.write("\n\n----------------------------------------\n\n")


# Summary
avg_time = total_runtime / image_count if image_count else 0

with open(total_txt_path, "a", encoding="utf-8") as total_f:
    total_f.write("\nSUMMARY\n")
    total_f.write("=======\n")
    total_f.write(f"Total images processed: {image_count}\n")
    total_f.write(f"Total runtime: {total_runtime:.4f} seconds\n")
    total_f.write(f"Average per image: {avg_time:.4f} seconds\n")

print(f"\nAll results saved to: {total_txt_path}")
