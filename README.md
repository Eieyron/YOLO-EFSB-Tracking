"# YOLO-EFSB-Tracking" 

Install the dependencies:
 - tensorflow 1.13.1
 - opencv 4.1.2
 - numpy 1.17.1

1) To be able to run the program, clone this repository to your computer system.

2) you will need to download 3 weights files:

https://drive.google.com/drive/folders/1wbnJeq8JHnvQNG50M8nxyRKvc8pjTYVA
in the weights folder of this google drive folder
 - copy the contents of the model data folder to the model data folder of your local repository
 - copy the contents of logs-000 to the YOLO-EFSB-Tracking/logs/000 folder of your local repository

3) Copy any video from the sample input video folder of the google drive folder to the same directory as final_project.py

4) to run the program, run the command 
	$python final_project.py <video_name>

5) let the AI process the whole video file, and 3 output files will be produced: an output.avi, an Image Trajectory file and a log_latest_output.txt


