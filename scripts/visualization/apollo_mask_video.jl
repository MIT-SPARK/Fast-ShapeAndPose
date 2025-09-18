## View Apollo keypoints for debugging
# you have to run nocs.jl first
# Lorenzo Shaikewitz, 9/4/2025

import Images
import GeometryBasics
import FileIO
import JSON
using Serialization
using Glob
using FastPACE
import Plots
using SimpleRotations
using Printf
using LinearAlgebra

parentimg = "/home/lorenzo/Downloads/apolloscape/car-instance/3d-car-understanding-train/train/images/"
object = "camera"
save_dir = "data/apollo_video_estc"

# load data
dets = JSON.parsefile("data/apolloscape2/robin_detections.json")
data = deserialize("data/apolloscape2/SCF.dat")

id2name = JSON.parsefile("data/apolloscape2/car_ids.json")
# ids = sort(parse.(Int,collect(keys(id2name))))
# keys_shapes_car_names = id2name.(string.(ids))

kpts_all = Dict()
gt_all = Dict()
robin_all = Dict()
for (img_name,dets_img) in dets
    kpts_all[img_name] = []
    gt_all[img_name] = []
    robin_all[img_name] = []
    for d in dets_img
        push!(kpts_all[img_name], convert.(Float64, reduce(hcat,d["est_world_keypoints"]))') # [m]
        T = convert.(Float64,reduce(hcat,d["gt_pose"]))'
        gt_shape = Int(d["car_id"])
        push!(gt_all[img_name], (project2SO3(T[1:3,1:3]), T[1:3,4], gt_shape))
        # robin
        push!(robin_all[img_name], Dict("time"=>d["robin_time"], "inliers"=>d["robin_inliers"]))
    end
end

cadpaths = "/home/lorenzo/Downloads/apolloscape/pku-autonomous-driving/car_models_json/"

R_fix = axang2rotm([0;0;1], π)#*axang2rotm([0;1;0], π)

"""
Get segmentation mask by tracing object pose
"""
function get_mask(img, R, t, cad_m, camK)
    # get bbox using vertices
    bbox = zeros(Int,2,2) # [min u max u; min v max v]
    bbox[:,1] = [1000;1000]
    mask = zeros(Bool, size(img))
    for coord in GeometryBasics.coordinates(cad_m)
        pixel = camK*(R*coord + t)
        pixel ./= pixel[3]
        coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
        coords[1] = clamp(coords[1], 1, size(img)[2])
        coords[2] = clamp(coords[2], 1, size(img)[1])

        bbox[1,1] = min(bbox[1,1], coords[1])
        bbox[1,2] = max(bbox[1,2], coords[1])
        bbox[2,1] = min(bbox[2,1], coords[2])
        bbox[2,2] = max(bbox[2,2], coords[2])

        mask[coords[2], coords[1]] = true
    end

    # ray casting helper function: check intersection with triangle
    function check_triangle(face, coords, ray_origin, ray_direction)
        triangle = coords[face]

        # face normal
        e1 = triangle[2] - triangle[1]
        e2 = triangle[3] - triangle[1]

        # weights in Barycentric coordinates
        q = ray_direction × e2
        a = e1'*q
        
        s = ray_origin - triangle[1]
        r = s × e1
        weights = [0; s'*q/a; ray_direction'*r/a]
        weights[1] = 1. - (weights[2]+weights[3])
    
        dist = e2'*r/a
    
        ϵ = 1e-7
        if (a <= ϵ) || sum(weights .< -ϵ) > 0 || (dist <= 0)
            return false
        else
            return true
        end
    end

    # transform coords by pose
    coords = copy(GeometryBasics.coordinates(cad_m))
    for (i,c) in enumerate(coords)
        coords[i] = R*c + t
    end
    faces = GeometryBasics.faces(cad_m)

    # build mask
    for i in bbox[1,1]:bbox[1,2]
        for j in bbox[2,1]:bbox[2,2]
            if mask[j,i]
                continue
            end

            # compute ray direction
            ray_direction = normalize(inv(camK)*[i;j;1])
            ray_origin = zeros(3)

            # check intersection with face
            for face in faces
                if check_triangle(face, coords, ray_origin, ray_direction)
                    mask[j, i] = true
                    break
                end
            end
        end
    end
    return mask
end

"""
Get lazy mask by projecting vertices of object
"""
function get_lazy_mask(img, R, t, cad_m, camK)
    mask = zeros(Bool, size(img))
    for coord in GeometryBasics.coordinates(cad_m)
        pixel = camK*(R*coord + t)
        pixel ./= pixel[3]
        coords = [Int(round(pixel[1])), Int(round(pixel[2]))]
        coords[1] = clamp(coords[1], 1, size(img)[2])
        coords[2] = clamp(coords[2], 1, size(img)[1])
        mask[coords[2], coords[1]] = true
    end
    return mask
end

function plot_mask!(plt, img, cad_m, pose, camK; lazy=true)
    R = pose[1]
    t = pose[2]
    if lazy
        # just project vertices
        mask = get_lazy_mask(img, R, t, cad_m, camK)
    else
        # dense segmentation mask
        mask = get_mask(img, R, t, cad_m, camK)
    end
    img_mask = Images.RGBA.(copy(img)).*0
    # img_mask[mask] .= Images.RGBA(0,1,1, 0.5)
    # img_mask[mask] .= Images.RGBA(1,0.5,0, 0.5)
    img_mask[mask] .= Images.RGBA(1,0,1, 0.5)
    Plots.plot!(img_mask, grid=false, axis=false)
    return plt, mask
end

"""
Plot outline of object on image.
"""
function plot_outline!(plt, img, cadpath, object_id, pose, camK)
    # get segmentation mask
    cad = FileIO.load(cadpath*(@sprintf "obj_%06d.ply" object_id))
    cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad)/1000, cad.faces)

    seg = get_mask(img, pose[1], pose[2], cad_m, camK)
    plot_outline!(plt, img, seg)
    return plt, seg
end

function plot_outline!(plt, img, seg)
    bound = boundary_map(BitMatrix(seg))
    img_new = Images.RGBA.(copy(img)).*0
    img_new[bound] .= Images.RGBA(0,0.5,0.5, 1.)
    # img_new[bound] .= Images.RGBA(0.5,0.25,0., 1.)
    Plots.plot!(plt, img_new)
    return plt
end

# stolen from https://github.com/lucianolorenti/ImageSegmentationEvaluation.jl/blob/master/src/utils.jl#L61
function boundary_map(seg::BitMatrix)
    local ee = zeros(size(seg));
    local s = zeros(size(seg));
    local se = zeros(size(seg));

    ee[:,1:end-1] = seg[ :,2:end];
    s[1:end-1,:] = seg[2:end,:];
    se[1:end-1,1:end-1] = seg[2:end,2:end];

    local b = (seg.!=ee) .| (seg.!=s) .| (seg.!=se);
    b[end,:] = seg[end,:] .!= ee[end,:];
    b[:,end] = seg[:,end] .!= s[:,end];
    b[end,end] = 0;
    return b
end


println("Starting $(length(collect(keys(data["solns"])))) frames...")


K = [2304.5479 0. 1686.2379; 0 2305.8757 1354.9849; 0 0 1]
K_scaled = [K[1:2,:] / 5; 0 0 1]

for (i,img_name) in enumerate(sort(collect(keys(data["solns"]))))

    # plot image
    img = Images.load(parentimg*"/"*img_name*".jpg")
    img = Images.imresize(img, ratio=1/5) # to make this tractable
    plt = Plots.plot(img, axis=false, grid=false, title="$img_name")

    solns_cur = data["solns"][img_name]
    for id in 1:length(keys(data["solns"][img_name]))
        # y = convert.(Float64,reduce(hcat, dets[img_name][id]["est_pixel_keypoints"])) ./ 5


        # plot keypoints
        # for (idx, kpt) = enumerate(eachcol(y))
        #     u = Int(round(kpt[2]))
        #     v = Int(round(kpt[1]))
        #     Plots.scatter!(plt, [v], [u], ms=1, label=false, c=:gold, msw=0)
        # end


        # ground truth
        pose_gt = gt_all[img_name][id]
        # shape
        # c_gt = pose_gt[3]
        # cad = JSON.parsefile(cadpaths*"/$(id2name(string(c_gt))).json")
        # faces = GeometryBasics.GLTriangleFace.(cad["faces"])
        # vertices = GeometryBasics.Point{3, Float32}.(cad["vertices"])
        # cad_m = GeometryBasics.Mesh(vertices, faces)
        # pose
        pose_gt = (pose_gt[1]*R_fix, pose_gt[2])

        # pose estimate
        soln = solns_cur[id]
        c_est = argmax(soln.c[:,1]) - 1
        cad = JSON.parsefile(cadpaths*"/$(id2name(string(c_est))).json")
        faces = GeometryBasics.GLTriangleFace.(cad["faces"])
        vertices = GeometryBasics.Point{3, Float32}.(cad["vertices"])
        cad_m = GeometryBasics.Mesh(vertices, faces)

        # plot mask and outline
        (plt, seg) = plot_mask!(plt, img, cad_m, (soln.R*R_fix, soln.p), K_scaled; lazy=false)
        # (plt, seg) = plot_mask!(plt, img, cad_m, pose_gt, K_scaled; lazy=false)
        plot_outline!(plt, img, seg)

    end

    Plots.savefig(save_dir*"/$(img_name).svg")

    print("$i ")
end

