from paddlex import create_model
import os

# Load your pedestrian attribute model
model = create_model(
    model_name="PP-LCNet_x1_0_pedestrian_attribute",
    model_dir="PP-LCNet_x1_0_pedestrian_attribute_infer"
)

image_dir = "./image"
output_dir = "./output"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Loop through everything in ./image
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)

    # Skip non-image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        print(f"Skipping non-image file: {filename}")
        continue

    print(f"Processing: {img_path}")

    # Predict (returns a generator)
    for res in model.predict(img_path):
        res.print()                      # print result
        res.save_to_img(output_dir)      # save visualization
        res.save_to_json(output_dir)     # optional JSON output
