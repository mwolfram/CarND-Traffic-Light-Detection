# get dependencies
pip install tqdm
pip install moviepy
sudo apt-get install ffmpeg

# get project
git clone https://github.com/mwolfram/CarND-Traffic-Light-Detection.git
cd CarND-Traffic-Light-Detection

# prep kitti data
#cd data
#wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
#unzip data_road.zip

# get test video
#mkdir test_videos
#cd test_videos
#wget https://www.dropbox.com/s/9iiy7u4nv5l77ww/hart1.mp4?dl=1

#cd ..
#cd ..

# run project
python main.py
