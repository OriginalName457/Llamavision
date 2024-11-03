import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, Button, Entry
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import os
import glob

# Check if GPU is available
gpu_status = "GPU is engaged!" if torch.cuda.is_available() else "Failed to engage GPU. Running on CPU."

# File paths and variables for temporary files and image management
memory_file_path = "temp_chat_memory.txt"
temp_image_dir = "temp_images"
image_counter = 0  # Track how many images are uploaded
image_history = []  # Track the history of uploaded images

# Ensure the temp image directory exists
if not os.path.exists(temp_image_dir):
    os.makedirs(temp_image_dir)

# Load the LLaMA 3.2 Vision Instruct Model and Processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Instruction prompt to ensure logical ending and prevent repetition
instruction_prompt = """
Describe the objects, setting, and appearance of the people or elements in the image in detail, but without repeating information. Ensure the response finishes logically, with a clear beginning and end to the response.
Limit the response to 2-3 paragraphs and end with: 'Does this satisfy your request, or would you like to know more?'
"""

# Function to load the model and processor
def install_model():
    try:
        print("Loading LLaMA 3.2-Vision Instruct model...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # Ensure model weights are tied to remove the warning
        model.tie_weights()

        print("Model and processor loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None, None

# Load the model and processor
model, processor = install_model()

# Function to write chat history to memory
def write_to_memory(content):
    with open(memory_file_path, "a") as f:
        f.write(content + "\n")

# Function to read the chat memory
def read_from_memory():
    if os.path.exists(memory_file_path):
        with open(memory_file_path, "r") as f:
            return f.read()
    return ""

# Function to clear the chat memory
def clear_memory():
    global image_counter, image_history
    if os.path.exists(memory_file_path):
        os.remove(memory_file_path)
    image_counter = 0  # Reset image counter
    image_history.clear()  # Clear image history

# Function to clean up temp image files
def clean_up_temp_images():
    files = glob.glob(os.path.join(temp_image_dir, "*.jpg"))
    for f in files:
        os.remove(f)

# Function to handle image uploads
def load_image():
    global image_counter, image_history
    if not model or not processor:
        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, "Model not loaded. Check setup.\n")
        output_box.config(state=tk.DISABLED)
        return

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.png")]
    )
    if file_path:
        img = Image.open(file_path)

        # Convert RGBA to RGB if necessary (JPEG doesn't support RGBA)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Save the uploaded image as a temp image with a unique name
        image_counter += 1
        temp_image_path = os.path.join(temp_image_dir, f"temp_image_{image_counter}.jpg")
        img.save(temp_image_path)  # Save image

        # Track the uploaded image
        image_history.append(f"Image {image_counter}: {temp_image_path}")
        write_to_memory(f"Image {image_counter}: {temp_image_path}")

        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, f"Image {image_counter} uploaded successfully.\n")
        output_box.config(state=tk.DISABLED)

# Function to handle chat input and model generation
def chat():
    if not model or not processor:
        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, "Model not loaded. Check setup.\n")
        output_box.config(state=tk.DISABLED)
        return

    user_input = chat_entry.get()
    if not user_input:
        return  # No input provided by the user

    try:
        # Retrieve previous chat memory
        memory_context = read_from_memory()

        # If there is an image, include it in the prompt
        if image_counter > 0:
            latest_image_path = os.path.join(temp_image_dir, f"temp_image_{image_counter}.jpg")
            prompt = f"<|image|>\n{user_input}"  # Only include user input and image token
            image = Image.open(latest_image_path)

            # Process the image and text together
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        else:
            # If no image, just process the text
            prompt = f"{user_input}"  # Only user input
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)

        # Generate the response with control over length and coherence
        output = model.generate(**inputs, max_new_tokens=250, max_length=350)
        response = processor.decode(output[0], skip_special_tokens=True)

        # Store the conversation in memory
        write_to_memory(f"User: {user_input}\nAI: {response}")

        # Display the AI's response in a clean format
        final_response = f"{response}\n\nDoes this satisfy your request, or would you like to know more?"
        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, f"User: {user_input}\nLlama: {final_response}\n\n")
        output_box.config(state=tk.DISABLED)
        chat_entry.delete(0, tk.END)  # Clear the input box
    except Exception as e:
        output_box.config(state=tk.NORMAL)
        output_box.insert(tk.END, f"Error generating output: {e}\n")
        output_box.config(state=tk.DISABLED)

# Function to clear the chat and memory
def clear_chat():
    clear_memory()
    clean_up_temp_images()  # Clean up images
    output_box.config(state=tk.NORMAL)
    output_box.delete(1.0, tk.END)
    output_box.config(state=tk.DISABLED)

# Function to close the program and clean up
def close_program():
    clear_memory()
    clean_up_temp_images()  # Remove temp images
    root.quit()

# Set up the Tkinter interface
root = tk.Tk()
root.title("LLaMA ChatGPT-like Interface")
root.geometry("600x600")

# GPU status label
gpu_label = tk.Label(root, text=gpu_status, fg="green" if torch.cuda.is_available() else "red")
gpu_label.pack(pady=5)

# Output box for the chat
output_box = Text(root, wrap=tk.WORD, height=15)
output_box.config(state=tk.DISABLED)

# Scrollbar for the output box
scrollbar = Scrollbar(root, command=output_box.yview)
output_box.config(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Add the output box and scrollbar to the window
output_box.pack(expand=True, fill='both')

# Input box for user chat
chat_entry = Entry(root, width=50)
chat_entry.pack(pady=5)

# Send button to submit chat input
send_button = Button(root, text="Send", command=chat)
send_button.pack(pady=5)

# Button to upload an image
image_button = Button(root, text="Upload Image", command=load_image)
image_button.pack(pady=5)

# Button to clear chat
clear_button = Button(root, text="Clear Chat", command=clear_chat)
clear_button.pack(pady=5)

# Button to close the program
close_button = Button(root, text="Close", command=close_program)
close_button.pack(pady=5)

# Start the Tkinter main loop
root.mainloop()
