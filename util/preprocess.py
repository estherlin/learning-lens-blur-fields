import cv2
import json
import numpy as np
import rawpy

# Just some initializations for camera calibration
K = np.array(
    [[303, 0, 3120],
     [0, 303, 2090],
     [0, 0, 1]]
     , dtype=np.float32)
initial_dist_coeffs = np.zeros((5, 1), dtype=np.float32) # Assume no distortion

# Let's try homographies the old fashioned way
def apply_homography(src_centres, qry_centres, sharp_img, thresh=5):
    n = 1
    src_pts = [[i[0]/n, i[1]/n] for i in src_centres]
    src_pts = np.array(src_pts).astype(np.float32)
    query_pts = [(l[0]/n, l[1]/n) for l in qry_centres]
    query_pts = np.array(query_pts).astype(np.float32)
    print("src: ", src_pts.shape, "qry: ", query_pts.shape)

    # Find homographies    
    M, mask = cv2.findHomography(query_pts, src_pts, cv2.RANSAC, thresh)
    invM, invmask = cv2.findHomography(src_pts, query_pts, cv2.RANSAC, thresh)

    h,w = sharp_img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(query_pts, src_pts,M)

    # Warp the image
    out = cv2.warpPerspective(
        sharp_img,invM,(w, h),
        flags=(cv2.INTER_LINEAR), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue = [0,0,0,0])
    
    return out, M, invM, src_pts, query_pts

def get_lp_to_filenames(files):
    data_files = {}
    for file in files:
        if ".json" in file and "project" not in file:
            json_f = open(f"{file}")
            data = json.load(json_f)
            json_f.close()

            if data["requestedFrameSpec"]["lensPosition"] not in data_files:
                data_files[data["requestedFrameSpec"]["lensPosition"]] = [file.split(".")[0]]
            else:
                data_files[data["requestedFrameSpec"]["lensPosition"]].append(file.split(".")[0])
    return data_files

def raw_to_rgb(data_files):
    avg_data = {}
    for lp, files in data_files.items():
        avg_data[lp] = np.zeros((1512, 2016, 3))
        for file in files:
            with rawpy.imread(f"{file}.DNG") as raw:
                img = raw.raw_image.copy()
                avg_data[lp][:,:,0] += img[::2,::2]
                avg_data[lp][:,:,1] += img[::2,1::2]
                avg_data[lp][:,:,1] += img[1::2,::2]
                avg_data[lp][:,:,2] += img[1::2,1::2]
        avg_data[lp] /= len(files)
        avg_data[lp][:,:,1] /= 2

        # Remove black level
        avg_data[lp] -= 2112

    # Convert into 3D array
    data = np.zeros((len(avg_data.keys()), 1512, 2016, 3))
    for i, lp in enumerate(avg_data.keys()):
        data[i,...] = avg_data[lp]
    return data


def warp_sharp_to_lp(sharp_img, invM):
    """ Use invM to warp sharp image to lens position """
    h,w = sharp_img.shape
    # Warp the image
    out = cv2.warpPerspective(
        np.float32(sharp_img),np.float32(invM),(w, h),
        flags=(cv2.INTER_LINEAR), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue = [0,0,0,0])
    return out

def warp_lp_to_sharp(blur_img, M):
    """ Use M to warp lens position to sharp image """
    h,w = blur_img.shape
    # Warp the image
    out = cv2.warpPerspective(
        blur_img,M,(w, h),
        flags=(cv2.INTER_LINEAR), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue = [0,0,0,0])
    return out


def convert_uint8(im):
	return np.uint8(255* np.abs(im)/np.abs(im).max())


def load_data(DATA_DIR, distances, patterns, lens_positions, burst_size):
    # Read in all rggb data
    data = np.zeros((len(distances), len(patterns), len(lens_positions), 1512, 2016, 4))
    for i, dist in  enumerate(distances):
        for j, pat in enumerate(patterns):
            for k, lp in enumerate(lens_positions):
                for b in range(burst_size):
                    filename = 2*k + b
                    with rawpy.imread(f"{DATA_DIR}{dist}/{pat}/{filename}.dng") as raw:
                        img = raw.raw_image
                        data[i,j,k,:,:,0] += img[::2,::2]
                        data[i,j,k,:,:,1] += img[::2,1::2]
                        data[i,j,k,:,:,2] += img[1::2,::2]
                        data[i,j,k,:,:,3] += img[1::2,1::2]
    # Divide by burst size and remove black level
    data /= burst_size
    data -= 2112

    # save data
    np.save(f'{DATA_DIR}calib/rgb-mean.npy', data)
    return data


def binarize(array1, array2):
    binary_array = (array1 > array2).astype(np.uint8)
    return binary_array


def detect_circle_centers_with_subpixel_accuracy(image, dp, minDist, param1, param2, minRadius, maxRadius, neighborhood_size, neighborhood=True):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    print("detected circles", circles.shape)
    # return circles[0,...]

    def refine_centroid(gray, center, radius, neighborhood_size):
        # Define the local neighborhood around the detected circle center
        x_min, x_max = int(np.floor(center[0] - neighborhood_size)), int(np.ceil(center[0] + neighborhood_size))
        y_min, y_max = int(np.floor(center[1] - neighborhood_size)), int(np.ceil(center[1] + neighborhood_size))

        # Extract the intensity values within the neighborhood
        neighborhood_intensity = gray[y_min:y_max, x_min:x_max]

        # Generate coordinate grids for the neighborhood
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

        # Compute the centroid coordinates using intensity-weighted localization
        centroid_x = np.sum(neighborhood_intensity * x_grid) / np.sum(neighborhood_intensity)
        centroid_y = np.sum(neighborhood_intensity * y_grid) / np.sum(neighborhood_intensity)
        return centroid_x, centroid_y
    
    if neighborhood and circles is not None:
        refined_circles = []
        for circle in circles[0]:
            center, radius = circle[:2], circle[2]
            
            refined = False
            while not refined:
                try:
                    centroid_x, centroid_y = refine_centroid(gray, center, radius, neighborhood_size)
                    # Add the refined circle to the list
                    refined_circles.append([centroid_x, centroid_y, radius])
                    refined = True
                except:
                    neighborhood_size -= 20
                    if neighborhood_size < 0:
                        refined = True
                        refined_circles.append([center[0], center[1], radius])
                    #refined_circles.append([center[0], center[1], radius])

        # Convert the list of refined circles to a NumPy array
        refined_circles = np.array(refined_circles)
        return refined_circles

    elif circles is not None:
        return circles[0,...]
    else:
        return None

def sort_circles(circles, num_rows, num_cols):
    # sort detected circles
    temp = circles[:, :3].astype(np.float32)
    ind = np.lexsort((temp[:,0],temp[:,1])) 
    temp = temp[ind][...,:2]
    temps = [temp[(num_cols)*i:(num_cols)*(i+1)] for i in range(num_rows)]
    inds = [np.lexsort((temps[i][:,1],temps[i][:,0])) for i in range(num_rows)]
    sorted_inds = np.concatenate([temps[i][inds[i]] for i in range(num_rows)], axis=0)
    return sorted_inds

def calibrate_camera(obj_points, img_points, width, height, ):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), K, initial_dist_coeffs)
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
    return ret, mtx, dist, rvecs, tvecs

def undistort_radial(img, camera_matrix, dist_coeffs):
    # Convert the 2D image to a 3-channel image
    img_3channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Undistort the 3-channel image
    undistorted_img_3channel = cv2.undistort(img_3channel, camera_matrix, dist_coeffs)

    # Convert the undistorted 3-channel image back to a single channel
    undistorted_img = cv2.cvtColor(undistorted_img_3channel, cv2.COLOR_BGR2GRAY)

    return undistorted_img

def reprojection_error(original_coords, transformed_coords, homography_matrix):
    original_coords_h = np.column_stack((original_coords, np.ones((original_coords.shape[0], 1))))
    reprojected_coords = (homography_matrix @ original_coords_h.T).T
    reprojected_coords = (reprojected_coords / reprojected_coords[:, 2, np.newaxis])[:, :2]
    error = np.sqrt(np.sum((reprojected_coords - transformed_coords) ** 2, axis=1))
    mean_error = np.mean(error)
    std_error = np.std(error)
    return mean_error, std_error, reprojected_coords

def norm_img(img):
    return np.float32(img/np.max(img))

def distort_points(points, camera_matrix, dist_coeffs):
    """Distort points using the camera matrix and distortion coefficients."""
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()

    x = (points[..., 0] - cx) / fx
    y = (points[..., 1] - cy) / fy

    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6)

    x_distorted = x / radial
    y_distorted = y / radial

    points_distorted = np.zeros(points.shape)
    points_distorted[..., 0] = x_distorted * fx + cx
    points_distorted[..., 1] = y_distorted * fy + cy

    return points_distorted