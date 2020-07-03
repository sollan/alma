import numpy as np
import pandas as pd
import cv2


file = './ladder rung results/Irregular_362_3dpi_croppedDLC_resnet50_Ladder RungMay12shuffle1_500000.csv'
video = './ladder rung results/Irregular_362_3dpi_cropped.avi'

vidcap = cv2.VideoCapture(video)

bodypart = 'HR'
data, filename = read_file(file)
data = fix_column_names(data)
data = filter_predictions(data, bodypart, 0)
start = 0
end = len(data)
end


id_frame = 675
threshold = 270
vidcap.set(1, id_frame)
ret, frame = vidcap.read()
color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

condition = (data[bodypart + ' y'] >= threshold) & (data[bodypart + ' y'] <= frame.shape[0])

# plot_data(data, bodypart, start, end, filename, 'y', 0)
plt.figure(figsize=(20,5))
plt.scatter(data['bodyparts coords'][condition]/end*frame.shape[1], data[bodypart + ' y'][condition], s = 30, color='b')
plt.scatter(data[bodypart + ' x'][id_frame], data[bodypart + ' y'][id_frame], s = 50, color='r')

# plt.scatter(id_frame, data[bodypart + ' y'][id_frame], s = 50, color = 'y')
# plt.figure(figsize=(20,5))
plt.imshow(color)
plt.scatter(id_frame/end*frame.shape[1], data[bodypart + ' y'][id_frame], s = 50, color = 'y')

# plt.xlim((start, end))
plt.title('frame %i (%d s)' %(id_frame, round(id_frame/119)))
plt.show()