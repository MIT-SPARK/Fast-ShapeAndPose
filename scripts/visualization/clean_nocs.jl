using JSON
using MAT
using glob

dets = JSON.parsefile("/home/lorenzo/Downloads/laptop-1/laptop-1/scene_1-laptop-every_category_keypoints.json")
dets = dets[1]
delete!(dets, "obj_000014")
delete!(dets, "obj_000008")
delete!(dets, "obj_000007")

shapes = zeros(3, length(dets[collect(keys(dets))[1]]),length(dets)) # 3 x N x K

for (i,obj) in enumerate(dets)
    obj = obj[2]
    shape = reduce(hcat,obj) # 3 x N
    shapes[:,:,i] = shape
end

# # file = matopen("data/nocs/laptop_tim/shapes.mat")
# matwrite("data/nocs/laptop_tim/shapes.mat", Dict("shapes"=>shapes))
# # close(file)

file = matopen("data/nocs/laptop_tim/shapes.mat", "w")
write(file, "shapes", shapes)
close(file)

# to count
bottle_ct = 0
camera_ct = 0
mug_ct = 0
laptop_ct = 0
nocs_loc = "/home/lorenzo/research/playground/catkeypoints/NOCS/real_test"
for folder in readdir(nocs_loc)
    for meta in glob("*_meta.txt",nocs_loc*"/"*folder)
        lines = readlines(meta)
        lines = reduce(hcat, split.(lines, " "))
        ids = parse.(Int,lines[2,:])
        global bottle_ct += sum(ids .== 1)
        global camera_ct += sum(ids .== 3)
        global mug_ct += sum(ids .== 6)
        global laptop_ct += sum(ids .== 5)
    end
end
