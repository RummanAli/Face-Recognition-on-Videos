#from facenet_pytorch import  InceptionResnetV1
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
from mtcnn import MTCNN
from cosface import sphere20a
import argparse
def frame_to_video(image_array): 
    height, width, layers = image_array[0].shape
    size = (width,height)
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()


def face_annotation(img,detect,names,index):
    img = Image.fromarray(img)
    image = ImageDraw.Draw(img)
    if (len(detect) != 0 ):
        for i,boxe in enumerate(detect):
            x, y, width, height = boxe['box']
            shape = [(x, y), (x + width, height + y)]
            image.rectangle(shape,outline = "red",width=5)
            font = ImageFont.truetype("arial.ttf", 20)
            image.text((x, y), names[i],font = font)
            #img.save("ImageDraw"+str(index)+".png")
    return img


def embed_imgs(img_path,model,detector):
    #img_path = "images"
    emb_path = "embeddings"
    for filename in os.listdir(img_path):
        #resnet = InceptionResnetV1(pretrained='vggface2').eval()
        detector = MTCNN()
        img = np.array(Image.open(os.path.join(img_path,filename)))
        result = []
        result.append(detector.detect_faces(np.array(img))[0])
        #img_cropped = mtcnn(img)
        #if(len(result) == 2):
        
        img_cropped = face_alignment(result,img)
        #else:
        #    img_cropped = face_alignment(result,img)

        model = sphere20a()
        model.load_state_dict(torch.load('sphere20a_20171020.pth'))
        model.feature = True

        faces_cropped = np.array(img_cropped)
        #for i in range(faces_cropped):
        #   faces_cropped[i] = (faces_cropped[i]-127.5)/128
        #faces_cropped = np.expand_dims(faces_cropped,0)

        faces_cropped = np.rollaxis(faces_cropped,3,1)
        #img_embeddings = resnet(torch.tensor(faces_cropped).float())#.unsqueeze(0)[0])
        img_embeddings = model(torch.tensor(faces_cropped).float())#.unsqueeze(0)[0])
        
        
        img_embeddings = img_embeddings.detach().numpy()
        np.save(os.path.join(emb_path,filename[:-4]),img_embeddings)


def face_alignment(result,frame):
  #img = plt.imread(img_path)
  #detector = MTCNN()
  #result = detector.detect_faces(img)
    margin = 50
    scaled = []
    if (len(result)!=0):
        for box in result:
            bb = np.zeros(4, dtype=np.int32)
            x,y,width,height = box['box']
            bb[0] = np.maximum(x-margin/2, 0)
            bb[1] = np.maximum(y-margin/2, 0)
            bb[2] = np.minimum(width+margin/2, frame.shape[1])
            bb[3] = np.minimum(height+margin/2, frame.shape[0])
            cropped = frame[bb[1]:(bb[1]+bb[3]),bb[0]:bb[0]+bb[2],:]
            scaled.append((np.array((Image.fromarray(cropped)).resize((112, 96),Image.BILINEAR))-127.5)/128)
    else:
        scaled.append((np.array((Image.fromarray(frame)).resize((112, 96),Image.BILINEAR))-127.5)/128)
    return scaled




parser = argparse.ArgumentParser(description='PyTorch sphereface video')
parser.add_argument('--img', default='images', type=str)
parser.add_argument('--model','-m', default='sphere20a_20171020.pth', type=str)
parser.add_argument('--video','-v', default='sample.mp4', type=str)
parser.add_argument('--embed','-e', default='false', type=str)
args = parser.parse_args()   

detector = MTCNN()
model = sphere20a()
model.load_state_dict(torch.load(args.model))
model.feature = True

emb_path = "embeddings"
img_path = args.img
if(args.embed == "true"):
    embed_imgs(img_path,model,detector)
#resnet = InceptionResnetV1(pretrained='vggface2').eval()
#mtcnn = mtcn(keep_all=True)
v_cap = cv2.VideoCapture(args.video)
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
img_array = []



for count in range(v_len):
    
    # Load frame
    success, frame = v_cap.read()
    if not success:
        continue
        
    # Add to batch, resizing for speed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = Image.fromarray(frame)
    result = detector.detect_faces(frame)
    faces_cropped = []
    if(len(result)!=0):
        faces_cropped = face_alignment(result,frame)
    #faces_cropped = mtcnn(frame)
    
    
    if (len(faces_cropped) == 0):
        img_array.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        continue
    
    
    faces_cropped = np.array(faces_cropped)
    #faces_cropped = np.expand_dims(faces_cropped,0)
    faces_cropped = np.rollaxis(faces_cropped,3,1)
    
    
    img_embeddings = model(torch.tensor(faces_cropped).float())#.unsqueeze(0)[0])
    img_embeddings = img_embeddings.detach().numpy()
    sim = []
    for filename in os.listdir(emb_path):
        img2_embeddings = np.load(os.path.join(emb_path,filename))
        sim.append(cosine_similarity(img_embeddings,img2_embeddings))
    sim = np.array(sim)
    indexes = np.argmax(sim,0)
    recognized_names = []
    for index in indexes:
        recognized_names.append(str(os.listdir(emb_path)[index[0]][:-4]))
    img_array.append(cv2.cvtColor(np.array(face_annotation(frame,result,recognized_names,count)), cv2.COLOR_RGB2BGR))
frame_to_video(img_array)