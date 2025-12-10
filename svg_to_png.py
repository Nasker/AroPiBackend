import nocairosvg
import os
import argparse

class SvgToPng:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height

    def convert(self, input_path, output_name):
        nocairosvg.svg2png(url=input_path, write_to=output_name,
                         output_width=self.width, output_height=self.height)
        print(f"Converted: {input_path} -> {output_name}")
    
    def batch_convert(self, input_dir, output_dir):
        print("Batch converting SVG files to PNG...")
        print(f"input_dir: {input_dir}")
        print(f"output_dir: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        converted_count = 0
        for file in os.listdir(input_dir):
            if file.endswith(".svg"):
                input_path = os.path.join(input_dir, file)
                output_name = os.path.join(output_dir, file.replace(".svg", ".png"))
                self.convert(input_path, output_name)
                converted_count += 1
        
        print(f"\nTotal files converted: {converted_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert SVG files to PNG format')
    parser.add_argument('--width', type=int, default=256, help='Output width in pixels (default: 256)')
    parser.add_argument('--height', type=int, default=256, help='Output height in pixels (default: 256)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    convert_parser = subparsers.add_parser('convert', help='Convert a single SVG file to PNG')
    convert_parser.add_argument('input', help='Input SVG file path')
    convert_parser.add_argument('output', help='Output PNG file path')
    
    batch_parser = subparsers.add_parser('batch', help='Batch convert all SVG files in a directory')
    batch_parser.add_argument('input_dir', help='Input directory containing SVG files')
    batch_parser.add_argument('output_dir', help='Output directory for PNG files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    svg_to_png = SvgToPng(width=args.width, height=args.height)
    
    if args.command == 'convert':
        svg_to_png.convert(args.input, args.output)
    elif args.command == 'batch':
        svg_to_png.batch_convert(args.input_dir, args.output_dir)