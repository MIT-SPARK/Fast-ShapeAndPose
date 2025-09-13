## Convert ApolloCar3D dataset (pixel keypoints -> keypoints with depth)
#
# Lorenzo Shaikewitz, 9/13/2025

import JSON
import Images
using FileIO

path_keypoints = "/home/lorenzo/research/playground/nepv/resources/gsnet/datasets/apollo/annotations/apollo_val.json"
parent_rgb = "/home/lorenzo/Downloads/apolloscape/stereo/stereo_train_merged/cameras/" # .jpg
parent_depth = "/home/lorenzo/Downloads/apolloscape/stereo/stereo_train_merged/disparity/" # .png

dets = JSON.parsefile(path_keypoints)
dets_converted = Dict()

for (i,det) in enumerate(dets["annotations"])
    # pixel keypoints
    y_px = reshape(det["keypoints"], (3,66))

    # depth
    image_name = dets["images"][i]["file_name"]
    depth_raw = load(parent_depth*image_name.split(".")[1]*".png") ./ 200
    

    depth_map = load_disparity_to_depth(disp_im_path, stereo_params['Q'])
end
