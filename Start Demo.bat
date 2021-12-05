Rem This will allow users a shortcut in windows to run our program (Demo Video)
cd obj_det
start cmd /k python yolo.py --play_video==True --video_path="videos/push.mp4"
