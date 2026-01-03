 # Converts the videos to mp3 
# this page is the convertor of video formate to mp4 formate 
# using the ffmpeg which is a Algorithm based convertor which we have to install and used it
# this is the 


import os
import subprocess

files = os.listdir("videos")

print(files)

for file in files:
     if file.endswith(".mp4"):
          print(file)
     tutorial_number = file.split("#")[1].replace(".mp4", "")
     tutorial_name = file.split("#")[0]
     print("Tutorial number is: ",tutorial_number)
     print("Tutorial name is: ",tutorial_name)
     print(tutorial_number,tutorial_name)

     subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{tutorial_number}_{tutorial_name}.mp3"])     