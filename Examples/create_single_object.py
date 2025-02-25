import sys
import argparse
import gcv_utils

def main(input_image_path: str, output_object_path: str, verbose: bool = False) -> None:
    try:
        # Read the VTK image
        image = gcv_utils.read_vtkimage(image_path = input_image_path, verbose = verbose)
        # Process the image to create mesh
        polydata = gcv_utils.create_vtkpolydata_from_vtkImage(image = image, apply_gaussian = True)
        # Write the output mesh
        gcv_utils.write_object(polydata = polydata, file_path = output_object_path, verbose = verbose)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Convert a VTK image to an object.")
    parser.add_argument("input_image_path", type = str, help = "Path to the input image.")
    parser.add_argument("output_object_path", type = str, help = "Path to save the output object.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    main(input_image_path = args.input_image_path, output_object_path = args.output_object_path, verbose = args.verbose)
