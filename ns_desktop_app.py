import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import cv2
import numpy as np
import tensorflow as tf
import datetime

classification_model = tf.keras.models.load_model("ns_classification_model.h5")
highlighting_model = tf.keras.models.load_model("ns_highlighting_model.h5")

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def preprocess_image(image):
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)).convert('L')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)

def classify_tumor(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = classification_model.predict(img)
    p = np.argmax(p, axis=1)[0]
    if p == 0:
        return 'Glioma Tumor'
    elif p == 1:
        return 'No tumor detected'
    elif p == 2:
        return 'Meningioma Tumor'
    else:
        return 'Pituitary Tumor'

def highlight_mask(original_image, mask):
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    custom_colormap[:, 0, 2] = np.arange(256)
    highlighted_mask = cv2.applyColorMap(mask_resized, custom_colormap)
    highlighted_image = cv2.addWeighted(original_image, 0.7, highlighted_mask, 0.3, 0)
    return highlighted_image

def predict_mask(image_array):
    mask = highlighting_model.predict(image_array)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask[0]

def predict_and_highlight(image):
    image_array = preprocess_image(image)
    tumor_mask = predict_mask(image_array)
    original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    tumor_mask_cv = tumor_mask.astype(np.uint8)
    highlighted_image = highlight_mask(original_image_cv, tumor_mask_cv)
    return Image.fromarray(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)), tumor_mask

class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroScan App")
        self.root.configure(background='black')
        self.root.iconbitmap(default='icon.ico')
        self.root.state('zoomed')  

        self.title_label = tk.Label(root, text="", font=("Arial", 24), fg="#04747c", bg="black")
        self.title_label.pack(pady=(20, 10))

        self.frame = tk.Frame(root, bg="black")
        self.frame.pack()

        self.image_label = tk.Label(self.frame, bg="black")
        self.image_label.pack()

        self.import_img = ImageTk.PhotoImage(file="import.png")
        self.predict_img = ImageTk.PhotoImage(file="predict.png")
        self.restart_img = ImageTk.PhotoImage(file="restart.png")
        self.save_img = ImageTk.PhotoImage(file="save.png")

        self.import_button = tk.Button(root, image=self.import_img, command=self.import_image, bd=0, relief=tk.FLAT, highlightthickness=0)
        self.import_button.pack(pady=(100, 10))

        self.predict_button = tk.Button(root, image=self.predict_img, command=self.predict_result, bd=0, relief=tk.FLAT, highlightthickness=0)
        self.predict_button.pack(pady=(10, 20))

        self.restart_button = tk.Button(root, image=self.restart_img, command=self.restart_app, bd=0, relief=tk.FLAT, highlightthickness=0)
        self.restart_button.pack(pady=(10, 20))
        self.restart_button.pack_forget()

        self.save_button = tk.Button(root, image=self.save_img, command=self.save_screenshot, bd=0, relief=tk.FLAT, highlightthickness=0)
        self.save_button.pack(pady=(7, 20))
        self.save_button.pack_forget()

        self.result_label = tk.Label(root, text="", font=("Arial", 18), fg="white", bg="black")
        self.result_label.pack(pady=10)

        self.danger_label = tk.Label(root, text="", font=("Arial", 18), fg="white", bg="black")
        self.danger_label.pack(pady=10)

        self.image = None
        self.load_placeholder_image()

    def load_placeholder_image(self):
        try:
            self.image = Image.open("sample.png")
            self.display_image(self.image)
        except Exception as e:
            print(f"Error loading placeholder image: {e}")

    def import_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.display_image(self.image)

    def display_image(self, image):
        resized_image = image.resize((300, 300))
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=self.tk_image)
        self.image_label.image = self.tk_image

    def predict_result(self):
        if self.image:
            classification = classify_tumor(self.image)
            self.result_label.configure(text=f"Prediction Result: {classification}")

            if "Tumor" in classification:
                highlighted_image, tumor_mask = predict_and_highlight(self.image)
                self.display_image(highlighted_image)

                highlighted_pixels = np.sum(tumor_mask > 0)
                total_pixels = tumor_mask.shape[0] * tumor_mask.shape[1]
                percentage_highlighted = (highlighted_pixels / total_pixels) * 100

                if percentage_highlighted < 3:
                    danger_level = "Low"
                    color = "green"
                elif percentage_highlighted < 7:
                    danger_level = "Medium"
                    color = "yellow"
                else:
                    danger_level = "High"
                    color = "red"

                self.danger_label.configure(text=f"Danger Level: {danger_level}", fg=color)
            else:
                self.danger_label.configure(text="Danger Level: Low", fg="green")

            self.import_button.pack_forget()
            self.predict_button.pack_forget()
            self.restart_button.pack(pady=(10, 20))
            self.save_button.pack(pady=(10, 20))

        else:
            messagebox.showerror("Error", "Please import an image first.")

    def save_screenshot(self):
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        width = x + self.root.winfo_width()
        height = y + self.root.winfo_height()
        screenshot = ImageGrab.grab().crop((x, y, width, height))
        
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"IMG_{current_date}.png"
        file_path = filedialog.asksaveasfilename(initialfile=filename,defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            screenshot.save(file_path)

    def restart_app(self):
        self.root.destroy()
        main()

def main():
    root = tk.Tk()
    app = TumorDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
