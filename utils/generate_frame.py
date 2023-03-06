import os
import glob
import cv2
from tqdm import tqdm

file = './video'

# movie = dir(fullfile(file,'*.AVI'));
if os.path.exists(file):
    movie_list = glob.glob(os.path.join(file, '*.AVI'))
else:
    print('cannot find file path')
    
# for num = 1:length(movie)
for movie_path in tqdm(movie_list):
    num = int(movie_path.split(os.sep)[-1].split('.')[0])
    # videofolder =  fullfile('annotation', ['0' movie(num).name(1:end-4)],'images');
    videofolder = os.path.join('./annotation', '{0:04}'.format(num),'images')
    # if( ~exist( videofolder, 'dir' ) ), mkdir( videofolder ), end;
    if not os.path.exists(videofolder):
        os.makedirs(videofolder)
    # obj = VideoReader(['video/' movie(num).name]);
    # numFrames = obj.NumberOfFrames;
    # for k = 1 : numFrames
    #     frame = read(obj,k);
    #     imwrite(frame, fullfile(videofolder,[num2str(k,'%04d') '.png']));
    # end
    cap = cv2.VideoCapture(movie_path)
    frame_num = 1
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(videofolder, '{0:04}.png'.format(frame_num)), frame)
            frame_num += 1
        else:
            break
    # %imshow(frame);
# %% following codes can be used for linux platform %% 
# %% install ffmpeg first, then fill video_path and annotation_path
# %     command = ['ffmpeg -i video_path/video/' movie(num).name '  annotation_path/' videofolder '/%04d.png'];
# %     system(command);
# end