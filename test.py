import math
MAX_FRAME = 10000
frames = []
frame_idx = 0
i = 0
while frame_idx < MAX_FRAME:
	frames.append(frame_idx)
	i += 1
	frame_idx += math.ceil(i/(10))

print(frames)
print(len(frames))
print(len(frames)/25)