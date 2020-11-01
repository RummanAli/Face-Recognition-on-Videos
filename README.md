# Face-Recognition-on-Videos
Recognizing faces in videos using SphereFace, OpenCV and MTCNN. Using the file system to store images and embeddings.

## Repo Info
- Pre-Trained Model: [Model](https://github.com/clcarwin/sphereface_pytorch/blob/master/model/sphere20a_20171020.7z)
- reference: [Sphereface pytroch](https://github.com/clcarwin/sphereface_pytorch)

## Usage
- Clone the repo.
- Download Pre-Trained Model
- Install MTCNN using **pip install mtcnn**
- Create two folders in local repo one named images and other named embeddings.
- Move images of faces to images folder
- Move the video for recognition to local repo.
- Activate the python environment
- Command to run the model: **python facerec-final.py --model sphere20a_20171020.pth --video "write name of video here" --embed true**

## Dependencies
- Pytroch
- Pillow
- Sklearn
- OpenCV
- MTCNN
- ArgParse

## Frame from output Video
![Recognized Image](https://github.com/RummanAli/Face-Recognition-on-Videos/blob/[main]/recognized_faces.jpg?raw=true)
