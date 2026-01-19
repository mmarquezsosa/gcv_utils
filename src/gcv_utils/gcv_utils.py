import os, vtk 
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from vtk.util import numpy_support
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

############################ SITK IMAGES ############################

def read_sitkimage(image_path: str, verbose: bool = False) -> sitk.Image:
    """
    Reads an sitkImage from the given file path.
    """
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
    
    try:
        image = sitk.ReadImage(image_path)
        if verbose:
            print(f"Image read from: {image_path}")
        return image
    
    except Exception as e:
        print(f"Error reading image: {e}")

def write_sitkimage(image: sitk.Image, image_path: str, verbose: bool = False) -> None:
    """
    Writes a vtkImageData to a file using an appropriate VTK writer based on the file extension.
    """ 
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Write the image with compression enabled.
        sitk.WriteImage(image, image_path, useCompression=True)
        if verbose:
            print(f"Image saved as: {image_path}")

    except Exception as e:
        print(f"Error saving image: {e}")

def convert_image_format(input_image_path: str, output_image_path: str, verbose: bool = False) -> None:
    """
    Converts an image from one format to another using SimpleITK.
    """
    try:
        image = read_sitkimage(input_image_path, verbose)
        write_sitkimage(image, output_image_path, verbose)

    except Exception as e:
        print(f"Error converting image format: {e}")

def create_sitkimage_from_array(image_array: np.ndarray, image_spacing: tuple[float, float, float], 
                                image_origin: tuple[float, float, float]) -> sitk.Image:
    """
    Converts a 3d numpy array to a SimpleITK image.
    """
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing(image_spacing)
    image.SetOrigin(image_origin)
    return image

def get_min_max_sitkimage(image: sitk.Image) -> tuple[float, float]:
    """
    Returns the minimum and maximum intensity values in an sitkImage.
    """
    try:
        min_max_filter = sitk.MinimumMaximumImageFilter()
        min_max_filter.Execute(image)
        return min_max_filter.GetMinimum(), min_max_filter.GetMaximum()
    except Exception as e:
        print(f"Error finding values: {e}")

def describe_detailed_sitkimage(image: sitk.Image, image_name: str = "Image", verbose: bool = False):
    """
    Returns metadata (dimensions, spacing, origin, direction, pixel type, and min-max) of an sitkImage.
    """
    try:
        image_size = image.GetSize()
        image_spacing = image.GetSpacing()
        image_origin = image.GetOrigin()
        image_direction = image.GetDirection()
        image_pixel_type = image.GetPixelIDTypeAsString()
        min_val, max_val = get_min_max_sitkimage(image)
        
        if verbose:
            info = (
                f"{image_name} Information:\n"
                f"  Size: {image_size}\n"
                f"  Spacing: {image_spacing}\n"
                f"  Origin: {image_origin}\n"
                f"  Direction: {image_direction}\n"
                f"  Pixel Type: {image_pixel_type}\n"
                f"  Min-Max:  ({min_val}, {max_val})\n"
            )
            print(info)
        
        return image_size, image_spacing, image_origin, image_direction, image_pixel_type, (min_val, max_val)
    
    except Exception as e:
        print(f"Error describing image: {e}")

def create_isotropic_spacing_from_min_spacing_sitkimage(image: sitk.Image) -> tuple[float, float, float]:
    """
    Returns isotropic spacing based on the minimum spacing in the sitkImage.
    """
    try:
        min_spacing = min(image.GetSpacing())
        return (min_spacing, min_spacing, min_spacing)
    
    except Exception as e:
        print(f"Error getting spacing: {e}")

def resample_sitkimage(image: sitk.Image, output_spacing: tuple[float, float, float], 
                       is_label: bool) -> sitk.Image:
    """
    Resamples an image to a given spacing while preserving size proportions.
    """
    try:
        input_spacing = image.GetSpacing()
        input_size = image.GetSize()

        # Calculate new size to maintain aspect ratio
        output_size = [
            int(round(input_size[i] * (input_spacing[i] / output_spacing[i])))
            for i in range(3)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(output_spacing)
        resample.SetSize(output_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())

        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
        default_pixel_value = 0 if is_label else -1024

        resample.SetDefaultPixelValue(default_pixel_value)
        resample.SetInterpolator(interpolator)

        return resample.Execute(image)
    
    except Exception as e:
        print(f"Error resampling image: {e}")

def resample_sitkimage_from_path(input_image_path: str, output_image_path:str, 
                                 output_spacing: tuple[float, float, float], is_label:bool, 
                                 verbose: bool = False) -> None:
    """
    Reads a sitkImage from file, resamples it using `resample_image`, and writes it.
    """
    try:
        image = read_sitkimage(input_image_path, verbose)
        resampled_image = resample_sitkimage(image, output_spacing, is_label)
        write_sitkimage(resampled_image, output_image_path, verbose)

    except Exception as e:
        print(f"Error resampling image: {e}")

def resample_and_center_sitkimage(image: sitk.Image, output_size: tuple[float, float, float], 
                            output_spacing: tuple[float, float, float], is_label: bool) -> sitk.Image:
    """
    Resizes and centers an image to a specific shape using padding and cropping.
    """
    # Define padding value depending on label vs. intensity image
    padding_value = 0 if is_label else -1064
    # First, resample the image to the desired output spacing
    processed_image = resample_sitkimage(image, output_spacing, is_label)

    # Calculate how much padding is needed
    resized_size = processed_image.GetSize()
    padding = [max(0, output_size[i] - resized_size[i]) for i in range(3)]
    lower_padding = [p // 2 for p in padding]
    upper_padding = [p - lp for p, lp in zip(padding, lower_padding)]

    # Apply constant padding
    padded_image = sitk.ConstantPad(processed_image, lower_padding, upper_padding, padding_value)

    # Check if cropping is needed (if we overshoot the target size)
    padded_size = padded_image.GetSize()
    cropping_needed = any(padded_size[i] > output_size[i] for i in range(3))

    if cropping_needed:
        crop_lower = [(padded_size[i] - output_size[i]) // 2 for i in range(3)]
        crop_upper = [padded_size[i] - output_size[i] - crop_lower[i] for i in range(3)]

        final_image = padded_image[
            crop_lower[0] : padded_size[0] - crop_upper[0],
            crop_lower[1] : padded_size[1] - crop_upper[1],
            crop_lower[2] : padded_size[2] - crop_upper[2]
        ]
    else:
        final_image = padded_image

    return final_image

def separate_binary_image_largest_components(image_path: str, verbose: bool = False):
    """
    Loads a 3D binary image, labels connected components, and selects:
      - The component with the smallest left centroid along the x-axis ("left")
      - The component with the largest right centroid along the x-axis ("right")
    """
    # Load the image (assumes a valid image format and that read_sitkimage is defined)
    binaryImage = read_sitkimage(image_path, verbose = verbose)

    # Label connected components
    labeled_image = sitk.ConnectedComponentImageFilter().Execute(binaryImage)

    # Compute label statistics (including bounding boxes)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled_image)

    labels = stats.GetLabels()
    if not labels:
        if verbose:
            raise ("No connected components found.")
        return None, None

    # Precompute statistics for each label: (num_voxels, centroid_x)
    label_stats = {}
    for label in labels:
        centroid = stats.GetCentroid(label)  # (x, y, z)
        num_voxels = stats.GetNumberOfPixels(label)
        label_stats[label] = (num_voxels, centroid[0])

    # Sort labels by number of voxels (descending) and take top 2
    top2_labels = sorted(label_stats.items(), key=lambda x: x[1][0], reverse=True)[:2]

    # Extract labels and centroid_x
    label1, (voxels1, cx1) = top2_labels[0]
    label2, (voxels2, cx2) = top2_labels[1]

    # Assign based on centroid_x
    if cx1 < cx2:
        right_candidate = label1
        left_candidate = label2
    else:
        right_candidate = label2
        left_candidate = label1

    # Create and save binary mask for the left candidate
    left_mask = sitk.BinaryThreshold(labeled_image,
                                     lowerThreshold=left_candidate,
                                     upperThreshold=left_candidate,
                                     insideValue=255,
                                     outsideValue=0)
    left_output_name = image_path[:-4] + "-Left.mhd"
    #write_sitkimage(image=sitk.Cast(left_mask, sitk.sitkInt16), image_path=left_output_name, verbose=verbose)

    # Create and save binary mask for the right candidate
    right_mask = sitk.BinaryThreshold(labeled_image,
                                      lowerThreshold=right_candidate,
                                      upperThreshold=right_candidate,
                                      insideValue=255,
                                      outsideValue=0)
    right_output_name = image_path[:-4] + "-Right.mhd"
    #write_sitkimage(image=sitk.Cast(right_mask, sitk.sitkInt16), image_path=right_output_name, verbose=verbose)
    
    return sitk.Cast(right_mask, sitk.sitkInt16), sitk.Cast(left_mask, sitk.sitkInt16)

############################ SITK & VTK IMAGES ############################

def get_array_from_image(image) -> np.ndarray:
    """
    Get a numpy array from the image.
    """
    if isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image)
    elif isinstance(image, vtk.vtkImageData):
        # Get the image dimensions: (width, height, depth)
        dims = image.GetDimensions()
        vtk_array = image.GetPointData().GetScalars()
        # Convert the flat VTK array to a NumPy array and reshape it to (depth, height, width)
        image_array = numpy_support.vtk_to_numpy(vtk_array).reshape((dims[2], dims[1], dims[0]))
    else:
        raise ValueError("Input image must be a sitk.Image Image or vtk.vtkImageData.")
    
    return image_array

def describe_image(image, image_name: str = "Image", verbose: bool = False):
    """
    Returns the dimensions, spacing, and origin of an image.
    """
    try:
        if isinstance(image, sitk.Image):
            image_size = image.GetSize()
            image_spacing = image.GetSpacing()
            image_origin = image.GetOrigin()
        elif isinstance(image, vtk.vtkImageData):
            image_size = image.GetDimensions()
            image_spacing = image.GetSpacing()
            image_origin = image.GetOrigin()
        else:
            raise TypeError("Input must be a sitk.Image or a vtk.vtkImageData.")

        if verbose:
            info = (
                f"{image_name} Information:\n"
                f"  Size: {image_size}\n"
                f"  Spacing: {image_spacing}\n"
                f"  Origin: {image_origin}\n"
            )
            print(info)

        return image_size, image_spacing, image_origin

    except Exception as e:
        print(f"Error describing image: {e}")

def plot_3D_views(image, image_name: str = "Image", display: bool = False,
                 save: bool = False, output_dir = None) -> None:
    """ 
    Plots the mid-slice views (sagittal, coronal, and axial) of a 3D image. 
    """
    volume = sitk.GetArrayFromImage(vol_sitk)

    sx, sy, sz = vol_sitk.GetSpacing()

    # Center slices
    z = volume.shape[0] // 2
    y = volume.shape[1] // 2
    x = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    # Axial
    axes[0].imshow(
        volume[z, :, :],
        cmap="gray",
        origin="lower",
        extent=[0, volume.shape[2] * sx, 0, volume.shape[1] * sy],
    )
    axes[0].set_title(f"Axial z={z}")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    # Coronal
    axes[1].imshow(
        volume[:, y, :],
        cmap="gray",
        origin="lower",
        extent=[0, volume.shape[2] * sx, 0, volume.shape[0] * sz],
    )
    axes[1].set_title(f"Coronal y={y}")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    # Sagittal
    axes[2].imshow(
        volume[:, :, x],
        cmap="gray",
        origin="lower",
        extent=[0, volume.shape[1] * sy, 0, volume.shape[0] * sz],
    )
    axes[2].set_title(f"Sagittal x={x}")
    axes[2].set_aspect("equal")
    axes[2].axis("off")

    plt.suptitle(image_name)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save:
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{image_name}.png")
        plt.savefig(path)
        print(f"Plot saved to {path}")

    if display:
        plt.show()
    else:
        plt.close()

############################ VTK IMAGES ############################

def read_vtkimage(image_path: str, verbose: bool = False) -> vtk.vtkImageData:
    """
    Reads a VTK image file from the given file path using an appropriate VTK reader based on the file extension.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        # Find file extension.
        extension = os.path.splitext(image_path)[1].lower()

        # Mapping file extensions to the corresponding VTK reader classes.
        reader_mapping = {
            '.mhd': vtk.vtkMetaImageReader,
            '.mha': vtk.vtkMetaImageReader,
            '.vti': vtk.vtkXMLImageDataReader,
            '.vtk': vtk.vtkDataSetReader,
            '.nii': vtk.vtkNIFTIImageReader,
            '.nii.gz': vtk.vtkNIFTIImageReader
        }

        reader_class = reader_mapping.get(extension)
        if reader_class is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        reader = reader_class()
        reader.SetFileName(image_path)
        reader.Update()
        image = reader.GetOutput()

        if verbose:
            print(f"Image read from: {image_path}")

        return image

    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def write_vtkimage(image: vtk.vtkImageData, image_path: str, verbose: bool = False) -> None:
    """
    Writes a vtkImageData to a file using an appropriate VTK writer based on the file extension.
    """
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the output directory exists
        
        # Determine the file extension
        extension = os.path.splitext(image_path)[1].lower()
        
        writer_mapping = {
            '.mhd': vtk.vtkMetaImageWriter,
            '.mha': vtk.vtkMetaImageWriter,
            '.vti': vtk.vtkXMLImageDataWriter,
            '.vtk': vtk.vtkDataSetWriter, 
            '.nii': vtk.vtkNIFTIImageWriter,
            '.nii.gz': vtk.vtkNIFTIImageWriter
        }
        writer_class = writer_mapping.get(extension)
        if writer_class is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        writer = writer_class()
        writer.SetFileName(image_path)
        writer.SetInputData(image)

        # Enable compression if supported by the writer
        if hasattr(writer, 'SetCompression'):
            writer.SetCompression(True)
        
        writer.Write()
        
        if verbose:
            print(f"Image saved as: {image_path}")
    
    except Exception as e:
        print(f"Error saving image: {e}")

def create_vtkimage_from_array(image_array: np.ndarray, image_spacing: tuple[float, float, float],
                               image_origin: tuple[float, float, float]) -> vtk.vtkImageData:
    """
    Convert a 3D numpy array to a vtkImageData object with short data type.
    """
    # Ensure the data is contiguous and convert to 16-bit integers (short)
    image_array = np.ascontiguousarray(image_array, dtype=np.int16)
    
    if image_array.ndim != 3:
        raise ValueError("Input image_data must be a 3D array.")
    image_dimensions = image_array.shape[::-1]  # reverse (z, y, x) to (x, y, z)

    # Flatten the data for VTK conversion
    flat_data = image_array.ravel()
    
    # Convert the numpy array to a VTK array with short data type
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True,
                                             array_type=vtk.VTK_SHORT)
    
    # Create vtkImageData and set its properties
    image = vtk.vtkImageData()
    image.SetDimensions(image_dimensions)
    image.SetSpacing(image_spacing)
    image.SetOrigin(image_origin)
    
    # Assign the converted data as the scalars of the image
    image.GetPointData().SetScalars(vtk_array)
    
    return image

############################ VTK POLYDATA ############################

def read_object(file_path: str, verbose: bool = False) :
    """
    Reads an VTK polydata from the given file path using an appropriate VTK reader based on the file extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Find file extension.
        extension = os.path.splitext(file_path)[1].lower()

        # Mapping file extensions to the corresponding VTK reader classes.
        reader_mapping = {
                        ".obj": vtk.vtkOBJReader,
                        ".stl": vtk.vtkSTLReader,
                        ".vtk": vtk.vtkPolyDataReader
        }

        reader_class = reader_mapping.get(extension)
        if reader_class is None:
            raise ValueError(f"Unsupported file format: {file_path}")

        reader = reader_class()
        reader.SetFileName(file_path)
        reader.Update()
        polydata = reader.GetOutput()

        if verbose:
            print(f"Object read from: {file_path}")
        return polydata

    except Exception as e:
        print(f"Error reading object: {e}")

def write_object(polydata: vtk.vtkPolyData, file_path: str, verbose: bool = False) -> None:
    """
    Writes vtkPolyData to a file using an appropriate VTK writer based on the file extension.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the output directory exists
        
        # Determine the file extension
        extension = os.path.splitext(file_path)[1].lower()
        
        writer_mapping = {
            ".obj": vtk.vtkOBJWriter,
            ".stl": vtk.vtkSTLWriter,
            ".vtk": vtk.vtkPolyDataWriter
        }
        writer_class = writer_mapping.get(extension)
        if writer_class is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        writer = writer_class()
        writer.SetFileName(file_path)
        writer.SetInputData(polydata)
        writer.Write()

        if verbose:
            print(f"Object saved as: {file_path}")

    except Exception as e:
        print(f"Error writing object: {e}")

def convert_object_format(input_file_path: str, output_file_path: str, verbose: bool = False) -> None:
    """
    Converts a mesh from one format to another.
    """
    try:
        polydata = read_object(input_file_path, verbose)
        write_object(polydata, output_file_path, verbose)

    except Exception as e:
        print(f"Error converting mesh: {e}")

def describe_vtkpolydata(polydata: vtk.vtkPolyData, polydata_name: str = "PolyData", 
                         verbose: bool = False):
    """
    Returns the number of points and cells in a vtkPolyData object.
    """
    try:
        num_points = polydata.GetNumberOfPoints()
        num_cells = polydata.GetNumberOfCells()
        
        if verbose:
            info = (
                f"{polydata_name} Information:\n"
                f"  Number of Points: {num_points}\n"
                f"  Number of Cells: {num_cells}\n"
            )
            print(info)

        return [num_points, num_cells]

    except Exception as e:
        print(f"Error describing polydata: {e}")

def get_array_from_vtkpolydata(polydata: vtk.vtkPolyData):
    """
    Convert vtkPolyData points and polygonal faces to numpy arrays.
    """
    try:
        num_points = polydata.GetNumberOfPoints()
        points = np.array([polydata.GetPoint(i) for i in range(num_points)])

        faces = []
        # Iterate over each cell in the polydata.
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            num_cell_points = cell.GetNumberOfPoints()
            # Only include cells with three or more points
            if num_cell_points >= 3:
                faces.append([cell.GetPointId(j) for j in range(num_cell_points)])
        
        return points, faces
    
    except Exception as e:
        print(f"Error getting array from polydata: {e}")


def create_vtkimage_from_vtkpolydata(polydata: vtk.vtkPolyData, image_size: tuple[int, int, int], 
                                     image_spacing: tuple[float, float, float], 
                                     image_origin: tuple[float, float, float]) -> vtk.vtkImageData:
    """
    Converts a closed vtkPolyData into a vtkImageData by using VTK's stencil conversion.
    """
    try:
        # Define image extent as (xmin, xmax, ymin, ymax, zmin, zmax)
        extent = (0, image_size[0] - 1,
                0, image_size[1] - 1,
                0, image_size[2] - 1)

        # Convert vtkPolyData to an image stencil.
        poly_to_stencil = vtk.vtkPolyDataToImageStencil()
        poly_to_stencil.SetInputData(polydata)
        poly_to_stencil.SetOutputOrigin(image_origin)
        poly_to_stencil.SetOutputSpacing(image_spacing)
        poly_to_stencil.SetOutputWholeExtent(extent)
        poly_to_stencil.Update()

        # Convert the image stencil to a vtkImageData.
        stencil_to_image = vtk.vtkImageStencilToImage()
        stencil_to_image.SetInputConnection(poly_to_stencil.GetOutputPort())
        stencil_to_image.SetInsideValue(255)  # Set voxels inside the surface.
        stencil_to_image.SetOutsideValue(0)   # Set voxels outside the surface.
        stencil_to_image.SetOutputScalarType(vtk.VTK_SHORT)
        stencil_to_image.Update()

        return stencil_to_image.GetOutput()
    
    except Exception as e:
        print(f"Error creating image from polydata: {e}")

def create_vtkpolydata_from_3Dcoordinates(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> vtk.vtkPolyData:
    """
    Create a VTK polydata object from 3D coordinate arrays.
    """
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z must have the same shape.")
    
    try:
        resolution_x, resolution_y = x.shape  # Extract grid resolution

        # Create vtkPoints object and allocate memory
        vtk_points = vtk.vtkPoints()
        num_points = resolution_x * resolution_y
        vtk_points.SetNumberOfPoints(num_points)

        # Assign coordinates to vtkPoints
        for idx, (px, py, pz) in enumerate(zip(x.ravel(), y.ravel(), z.ravel())):
            vtk_points.SetPoint(idx, px, py, pz)

        # Create a vtkCellArray for triangles
        triangles = vtk.vtkCellArray()

        # Construct triangle faces for the mesh
        for i in range(resolution_x - 1):
            for j in range(resolution_y - 1):
                p1 = i * resolution_y + j
                p2 = p1 + 1
                p3 = (i + 1) * resolution_y + j
                p4 = p3 + 1

                # First triangle (p1, p2, p3)
                triangles.InsertNextCell(3)
                triangles.InsertCellPoint(p1)
                triangles.InsertCellPoint(p2)
                triangles.InsertCellPoint(p3)

                # Second triangle (p2, p4, p3)
                triangles.InsertNextCell(3)
                triangles.InsertCellPoint(p2)
                triangles.InsertCellPoint(p4)
                triangles.InsertCellPoint(p3)

        # Create vtkPolyData and assign points and triangle faces
        vtk_polydata = vtk.vtkPolyData()
        vtk_polydata.SetPoints(vtk_points)
        vtk_polydata.SetPolys(triangles)

        return vtk_polydata
    
    except Exception as e:
        print(f"Error creating polydata: {e}")

def create_vtkpolydata_from_vtkImage(image: vtk.vtkImageData, apply_gaussian: bool = False) -> vtk.vtkPolyData:
    """
    Convert a vtkImageData to vtkPolyData using thresholding and Marching Cubes,
    optionally applying a Gaussian smoothing filter.
    """
    try:
        # Create a threshold filter
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(image)
        threshold.ThresholdByUpper(128)  # Set the threshold value
        threshold.Update() 

        # Determine which filter to use next: either direct threshold output or smoothed version
        if apply_gaussian:
            # Apply Gaussian smoothing to the thresholded image
            gaussian_smooth = vtk.vtkImageGaussianSmooth()
            gaussian_smooth.SetInputConnection(threshold.GetOutputPort())
            gaussian_smooth.SetStandardDeviations(1.0, 1.0, 1.0)  # Sigma values
            gaussian_smooth.Update()
            marching_input = gaussian_smooth.GetOutputPort()
        else:
            marching_input = threshold.GetOutputPort()

        # Create the Marching Cubes algorithm
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputConnection(marching_input)
        marching_cubes.SetValue(0, 128)  # Set the threshold value for surface extraction
        marching_cubes.Update()
        polydata=marching_cubes.GetOutput()

        return polydata # Return the resulting polydata (surface extracted by Marching Cubes)
    
    except Exception as e:
        print(f"Error creating polydata: {e}")


def plot_vtkpolydata(polydata: vtk.vtkPolyData, polydata_name: str = "Object", 
                     display: bool = False, save: bool = False, output_dir = None,
                     scale: bool = False) -> None:
    """
    Plot a vtkPolyData object using matplotlib
    """
    points, faces = get_array_from_vtkpolydata(polydata)

    # Create a new figure and a 3D subplot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection from the faces using the extracted points.
    poly_collection = Poly3DCollection([points[face] for face in faces],
                                         alpha=0.1, edgecolor='k')
    ax.add_collection3d(poly_collection)

    # Scatter plot the points.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=1)
    
    if scale:
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_zlim(points[:, 2].min(), points[:, 2].max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.suptitle(polydata_name, fontsize=16)    
    plt.tight_layout()

    if save:
        if output_dir is None:
            output_dir = os.getcwd()
        file_path = os.path.join(output_dir, f"{polydata_name}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        print(f"Plot saved to: {file_path}")

    if display:
        plt.show()
    else:
        plt.close()

def is_polydata_valid(polydata: vtk.vtkPolyData) -> bool:
    """
    Verify that a VTK polydata object is valid.
    
    A valid polydata must:
      - Not be None.
      - Have a non-empty points array.
      - Have at least one cell.
      - Contain no NaN values in its point coordinates.
      
    Returns:
        True if the polydata is valid, False otherwise.
    """
    # Check if the polydata object exists.
    if polydata is None:
        return False

    # Check if polydata has points.
    points = polydata.GetPoints()
    if points is None or points.GetNumberOfPoints() == 0:
        return False

    # Check if polydata has cells.
    if polydata.GetNumberOfCells() == 0:
        return False

    # Convert the VTK point data to a NumPy array for an efficient NaN check.
    point_array = numpy_support.vtk_to_numpy(points.GetData())
    if np.isnan(point_array).any():
        return False

    return True

def decimate_vtkpolydata(polydata: vtk.vtkPolyData, decimation_factor: float) -> vtk.vtkPolyData:
    """
    Reduce the number of points in a VTK PolyData given a factor using the Quadric Decimation filter.
    """
    if not (0.0 <= decimation_factor < 1.0):
        raise ValueError("Factor must be between 0.0 (no reduction) and 1.0 (nearly complete reduction).")
    try:
        decimation = vtk.vtkQuadricDecimation()
        decimation.SetInputData(polydata)
        decimation.SetTargetReduction(decimation_factor)
        decimation.Update()
        decimated_object = decimation.GetOutput()

        return decimated_object
    
    except Exception as e:
        print(f"Error decimating polydata: {e}")

def calculate_chamfer_distance_between_vtkpolydata(polydata1: vtk.vtkPolyData, polydata2: vtk.vtkPolyData) -> float:
    """
    Calculate the Chamfer distance between 2 vtkPolyData objects.
    """
    # Convert VTK point data to NumPy arrays
    points1 = numpy_support.vtk_to_numpy(polydata1.GetPoints().GetData())
    points2 = numpy_support.vtk_to_numpy(polydata2.GetPoints().GetData())

    if not is_polydata_valid(polydata1) or not is_polydata_valid(polydata2):
        raise ValueError("Invalid input polydata.")

    # Build KD-Trees for efficient nearest-neighbor search
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # For each point in polydata1, find the distance to its nearest neighbor in polydata2
    dists1, _ = tree2.query(points1)
    # For each point in polydata2, find the distance to its nearest neighbor in polydata1
    dists2, _ = tree1.query(points2)

    # Compute Chamfer loss as the sum of the mean squared distances in both directions
    loss = np.mean(dists1**2) + np.mean(dists2**2)
    return loss

def calculate_hausdorff_distance_between_vtkpolydata(polydata1: vtk.vtkPolyData, polydata2: vtk.vtkPolyData) -> float:
    """
    Calculate the Hausdorff loss between 2 vtkPolyData objects.
    """
    # Convert VTK point data to NumPy arrays
    points1 = numpy_support.vtk_to_numpy(polydata1.GetPoints().GetData())
    points2 = numpy_support.vtk_to_numpy(polydata2.GetPoints().GetData())

    if not is_polydata_valid(polydata1) or not is_polydata_valid(polydata2):
        raise ValueError("Invalid input polydata.")

    # Build KD-Trees for efficient nearest-neighbor search
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # For each point in polydata1, find the distance to its nearest neighbor in polydata2
    dists1, _ = tree2.query(points1)
    # For each point in polydata2, find the distance to its nearest neighbor in polydata1
    dists2, _ = tree1.query(points2)

    # Compute Hausdorff distance as the maximum of the directed distances
    hausdorff_distance = max(np.max(dists1), np.max(dists2))
    
    return hausdorff_distance
