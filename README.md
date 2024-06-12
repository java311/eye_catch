# Eye Catch

## Install
Just follow the install instrucctions of L2CS-net and Mivolo
- https://github.com/Ahmednull/L2CS-Net
- https://github.com/WildChlamydia/MiVOLO

First install L2CS-Net on conda. Then install MiVOlO on the same conda environment. 

The only fix, to make MiVOLO run, is the version of torchvision. 
Uninstall Torch vision  :
- <code> pip uninstall torch torchvision </code>

and then install this version :
-  <code> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  </code>


## Run
To run MiVOLO demo use the following line :
```
python demo.py --input "D:\faceswap\micro_cherry.mp4" --output "D:\faceswap\pinchi.mp4" --detector-weights "D:\tmp\eye_catch\MiVOLO\models\yolov8x_person_face.pt" --checkpoint "D:\tmp\eye_catch\MiVOLO\models\model_imdb_cross_person_4.22_99.46.pth.tar" --draw
```

To run L2CS demo just use: 
```
python demo.py
```

## Debug
How to set vscode python debug to use conda environment:
https://stackoverflow.com/questions/63411583/how-do-i-activate-my-conda-environment-for-vs-code-python-debugger-and-testing