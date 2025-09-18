## View NOCS keypoints for debugging
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

parentimg = "/home/lorenzo/research/playground/catkeypoints"
object = "camera"
save_dir = "data/nocs_video4"

# load data
det_files = glob("data/nocs/$object/robin_scene_*.json")
det_file = det_files[5]
dets = JSON.parsefile(det_file)
data = deserialize("data/nocs/$object/SCF.dat")

# cadpath = "data/nocs/mug/mug_anastasia_norm.obj"
# cadpath = "data/nocs/mug/mug_daniel_norm.obj"
# cadpath = "data/nocs/mug/mug_brown_starbucks_norm.obj"
cadpath = "data/nocs/camera/camera_canon_wo_len_norm.obj"

kpts_all = Dict()
gt_all = Dict()
robin_all = Dict()
Roffset = axang2rotm([1.;0;0],-π/2)
for file in det_files
    dets = JSON.parsefile(file)

    kpts_test = Dict()
    gt_test = Dict()
    robin_test = Dict()
    for (i,d) in enumerate(dets)
        kpts_test[i] = Dict(1=>convert.(Float64,reduce(hcat, d["est_world_keypoints"]))) # [m]
        T = convert.(Float64, reduce(hcat, d["gt_pose"]))'
        # gt_test[i] = Dict(1=>(T[1:3,1:3], T[1:3,4]))

        # fix the ground truth (to align models)
        # rotate R by 90 deg about x axis
        # Roffset = diagm(ones(3))

        gt_test[i] = Dict(1=>(project2SO3(T[1:3,1:3]), T[1:3,4]))

        # robin
        robin_test[i] = Dict("time"=>d["robin_time"], "inliers"=>d["robin_inliers"])
    end
    # push!(kpts_all, kpts_test)
    # push!(gt_all, gt_test)
    json = split(file,"/")[end]
    kpts_all[json] = kpts_test
    gt_all[json] = gt_test
    robin_all[json] = robin_test
end


if occursin("canon_len", cadpath)
    video_ref = "robin_scene_4-camera_canon_len_norm.json"
    frame_ref = 123
elseif occursin("canon_wo", cadpath)
    video_ref = "robin_scene_5-camera_canon_wo_len_norm.json"
    frame_ref = 242
elseif occursin("shengjun", cadpath)
    video_ref = "robin_scene_2-camera_shengjun_norm.json"
    frame_ref = 448
elseif occursin("mug_brown", cadpath)
    video_ref = "robin_scene_6-mug_brown_starbucks_norm.json"
    frame_ref = 103
elseif occursin("anastasia", cadpath)
    video_ref = "robin_scene_6-mug_anastasia_norm.json"
    frame_ref = 166
elseif occursin("mug_daniel", cadpath)
    video_ref = "robin_scene_2-mug_daniel_norm.json"
    frame_ref = 365
end
T1 = [data["solns"][video_ref][frame_ref].R data["solns"][video_ref][frame_ref].p; 0 0 0 1.]
T2 = [gt_all[video_ref][frame_ref][1][1] gt_all[video_ref][frame_ref][1][2]; 0 0 0 1.]

T_dif = T2*inv(T1)
R_dif = T_dif[1:3,1:3]
p_dif = T_dif[1:3,4]


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

function plot_mask!(plt, img, cadpath, pose, camK; lazy=true)
    # load CAD
    cad = FileIO.load(cadpath)
    cad_m = GeometryBasics.Mesh(GeometryBasics.coordinates(cad), cad.faces)

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
    img_mask[mask] .= Images.RGBA(0,1,1, 0.5)
    # img_mask[mask] .= Images.RGBA(1,0.5,0, 0.5)
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




println("Starting $(length(keys(data["solns"][split(det_file,"/")[end]]))) frames...")

for (i,imgframe) in enumerate(sort(collect(keys(data["solns"][split(det_file,"/")[end]]))))
    
    y = convert.(Float64,reduce(hcat, dets[imgframe]["est_pixel_keypoints"]))
    img_name = dets[imgframe]["rgb_image_filename"]

    # plot image
    img = Images.load(parentimg*"/"*img_name)
    plt = Plots.plot(img, axis=false, grid=false, title="$img_name")

    # plot keypoints
    for (idx, kpt) = enumerate(eachcol(y))
        u = Int(round(kpt[2]))
        v = Int(round(kpt[1]))
        Plots.scatter!(plt, [v], [u], ms=1, label=false, c=:gold, msw=0)
    end

    K = [591.0125 0. 322.525; 0 590.165 244.11084; 0 0 1]

    # ground truth
    pose_gt = gt_all[split(det_file,"/")[end]][imgframe][1]

    # pose estimate
    soln = data["solns"][split(det_file,"/")[end]][imgframe]

    # plot mask and outline
    (plt, seg) = plot_mask!(plt, img, cadpath, (soln.R*Roffset', soln.p), K; lazy=false)
    # (plt, seg) = plot_mask!(plt, img, cadpath, (R_dif*soln.R, R_dif*soln.p + p_dif), K; lazy=true)
    # (plt, seg) = plot_mask!(plt, img, cadpath, pose_gt, K; lazy=true)
    plot_outline!(plt, img, seg)

    Plots.savefig(save_dir*"/$(imgframe).svg")

    print("$i ")
end

