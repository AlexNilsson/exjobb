import bpy, math
from mathutils import Vector
import numpy as np

# Remove old
bpy.ops.object.select_by_type(type=='MESH')
bpy.ops.object.delete(use_global=False)

S = 10
P = 2

coords = []
relPositions = []


def getScaleAtDist(distance, frequency=1, phase=0):

    scaleFactor = 1/5

    s = math.sin(math.pi * frequency * distance + phase)

    return abs(s * scaleFactor)


for x in range (S):
    for y in range (S):
        for z in range (S):

            relPos = np.array([x, y, z]) / (S/2) - 1

            dist = np.sqrt(relPos.dot(relPos))

            if  dist <= 1:

                relPositions.append(dist)

                _x = x - S/2
                _y = y - S/2
                _z = z

                bpy.ops.mesh.primitive_cube_add(location=(_x,_y,_z))



# Create Animation

scene = bpy.context.scene


keyFrames = ()

scene.frame_set(1)

sets = 4
setFrames = 20

for i in range(sets):

    for i, obj in enumerate(scene.objects):
        s = getScaleAtDist(relPositions[i], frequency=i, phase=0*math.pi/2)
        obj.scale = (s,s,s)
        obj.keyframe_insert(data_path="scale", index=-1)

    scene.frame_set(i*setFrames)
