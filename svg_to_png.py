import cairosvg
import os
import tkinter
from tkinter import filedialog

class SvgToPng:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height

    def convert(self, input_path, output_name):
        cairosvg.svg2png(url=input_path, write_to=output_name,
                         output_width=self.width, output_height=self.height)
    def batch_convert(self, input_dir, output_dir):
        for file in os.listdir(input_dir):
            if file.endswith(".svg"):
                input_path = os.path.join(input_dir, file)
                output_name = os.path.join(output_dir, file.replace(".svg", ".png"))
                self.convert(input_path, output_name)

    def convert_gui(self):
        root = tkinter.Tk()
        root.withdraw()
        input_path = filedialog.askopenfilename(filetypes=[("SVG files", "*.svg")])
        output_name = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        self.convert(input_path, output_name)

    def batch_convert_gui(self):
        root = tkinter.Tk()
        root.withdraw()
        input_dir = filedialog.askdirectory()
        output_dir = filedialog.askdirectory()
        self.batch_convert(input_dir, output_dir)

if __name__ == "__main__":
    svg_to_png = SvgToPng()
    cmd = ''
    while cmd != 'q':
        cmd = input("Enter command: (c)onvert, (b)atch_convert, (q)uit:")
        if cmd == 'c':
            svg_to_png.convert_gui()
        elif cmd == 'b':
            svg_to_png.batch_convert_gui()
        elif cmd == 'q':
            break
    print("Goodbye!")