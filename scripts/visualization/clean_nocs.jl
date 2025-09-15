using JSON
using MAT

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

# file = matopen("data/nocs/laptop_tim/shapes.mat")
matwrite("data/nocs/laptop_tim/shapes.mat", Dict("shapes"=>shapes))
# close(file)