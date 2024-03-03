import base64
import requests

import cv2
import numpy as np

from gtts import gTTS
import os

import speech_recognition as sr

import threading
import queue
import time


#init

# OpenAI API Key
api_key = "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"

cv2.namedWindow("test")

r = sr.Recognizer() 

img_counter = 0

classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

filter=""

colors = np.random.uniform(0, 255, size=(len(classes), 3))
pb  = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
cvNet = cv2.dnn.readNetFromTensorflow(pb,pbt)   

searching = False

command = "nothing"



cam = cv2.VideoCapture(0)

ret, img = cam.read()

#it begins here >:D
   




# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


#listener
def listener(inputQueue):
  os.system("afplay bootsounds.mp3")
  os.system("afplay bootupmessage.mp3")


  while(True): 
    try:

      with sr.Microphone() as source2:
          
          r.adjust_for_ambient_noise(source2, duration=0.2)
          audio2 = r.listen(source2)
          
          # Using google to recognize audio
          MyText = r.recognize_google(audio2)
          MyText = MyText.lower()
          
          response = MyText
          print(response)
          if("alice" in response):
              if("find" in response):
                searching = True
                for elements in classes:

                   if(elements in response):
                      
                      global filter
                      filter = elements
                      break
                
              else:
                analyzeVideo(response)

              
          
            
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        
    except sr.UnknownValueError:
        print("unknown error occurred")

    inputQueue.put(audio2)



def analyzeVideo(prompt):
  ret, img = cam.read()
  os.remove("temp.png")
  img_name = "temp.png".format(0)

  cv2.imwrite(img_name, img) #THE PROBLEM IS HERE, IMG ORIGNALLY WAS SPAM REFRESHED HOW DO I TRANSFER IT FROM THERE
  
  os.system("afplay processingsound.mp3")

  print("thinking...")

  

  # name
  image_name = "temp.png"

  # Getting the base64 string
  base64_image = encode_image(image_name)

  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }

  payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt + ". Make any assumptions, keep responses to 2 sentences, and make sure you are descriptive to aid the visually impaired. Only refer the images only if need to"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 75
  }

  
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  content = response.json()["choices"][0]['message']['content']
  print(content)

  tts = gTTS(text=content, lang='en', slow=False)

  tts.save("speech.mp3")  # Save the generated speech as a temporary MP3 file
  os.system("afplay speech.mp3")  # Use the 'afplay' command to play the MP3 file





def main():
    global filter
    inputQueue = queue.Queue()

    inputThread = threading.Thread(target=listener, args=(inputQueue,), daemon=True)
    inputThread.start()

    cam = cv2.VideoCapture(0)



    while True:
        ret, img = cam.read()

        if (inputQueue.qsize() > 0):
            audio2 = inputQueue.get() 


        rows = img.shape[0]
        cols = img.shape[1]
        cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

        cvOut = cvNet.forward()

        for detection in cvOut[0,0,:,:]:
          score = float(detection[2])
          if score > 0.3:

            idx = int(detection[1])   # prediction class index. 


            if classes[idx] == filter or filter=="":
              if(classes[idx] == filter):
                analyzeVideo("Where is the " + filter)
                filter = ""

              left = detection[3] * cols
              top = detection[4] * rows
              right = detection[5] * cols
              bottom = detection[6] * rows
              cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                  

              label = "{}: {:.2f}%".format(classes[idx],score * 100)
              y = top - 15 if top - 15 > 15 else top + 15
              cv2.putText(img, label, (int(left), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
        cv2.imshow('my webcam', img)



        if not ret:
            print("failed to grab frame")
            break

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        time.sleep(0.01) 

if (__name__ == '__main__'): 
    main()


cam.release()

cv2.destroyAllWindows()








