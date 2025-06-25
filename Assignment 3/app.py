import tkinter as tk
import requests
import numpy as np
from PIL import Image, ImageDraw
from tkinter import messagebox


API_URL = "http://127.0.0.1:7000/predict"


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

        self.canvas_size = 680
        self.image_size = 28
        self.brush_size = 20

        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_button = tk.Button(self.button_frame, text="Predict Digit", command=self.predict_image)
        self.predict_button.pack(side="left")

        self.clear_button = tk.Button(self.button_frame, text="Erase", command=self.clear_canvas)
        self.clear_button.pack(side="right")

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)

        self.canvas.create_oval(x1, y1, x2, y2, fill="yellow", outline="yellow")

        scaled_x1, scaled_y1 = (x1 * self.image_size // self.canvas_size), (y1 * self.image_size // self.canvas_size)
        scaled_x2, scaled_y2 = (x2 * self.image_size // self.canvas_size), (y2 * self.image_size // self.canvas_size)
        self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill="yellow")

    def predict_image(self):
        image_data = np.array(self.image).reshape(-1) / 255.0
        image_list = image_data.tolist()

        try:
            response = requests.post(API_URL, json={"image": image_list})
            result = response.json()

            if "predicted_digit" in result:
                messagebox.showinfo("Result", f"Predicted Digit: {result['predicted_digit']}")
            else:
                messagebox.showerror("Error", "Failed to get prediction")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)


root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
