# Speed-Trap-2022 by Kaveen Jayamanna
## Bad training data
35526, 
49343, 
6851 
## Bad training data (retrain)
53284, 
35582, 
53311, 
8672, 
3022, 
33161, 
4513, 
35600, 
53162, 

## No keypoints annotation
27690,
51530,
40479,
50848,
51536 
## No keypoints  annotations (retrain)
7127, 
27532, 
27665, 
30061, 
151, 
27032, 
351, 
50063, 
482, 
51985, 
6248,  
33146, 
52008,
29133, 
3013, 
27636, 
32616, 
28238, 
50410

## Keypoints pre-processing
![image](https://user-images.githubusercontent.com/63982009/201197583-32cfa5cc-22d5-49c7-9fcb-9c6f05f3d2d6.png)
## Keypoints post-processing
![image](https://user-images.githubusercontent.com/63982009/201197852-4ddf7c6d-6b18-4082-9d14-92b4a31e2097.png)
## Speed algorithm
![image](https://user-images.githubusercontent.com/63982009/202248762-a7708299-9108-493a-9f60-9c32c7e1af1c.png)

## Inference Pipeline
![image](https://user-images.githubusercontent.com/63982009/201197122-ff1d82a5-b18c-4687-bcae-afbbefe84cf5.png)

## Demo: Before retraining
https://user-images.githubusercontent.com/63982009/200953267-8741dead-e932-4ac1-901c-52dd893336fb.mp4

## Demo: After retraining
https://user-images.githubusercontent.com/63982009/200953951-fbce303f-40e8-4416-a684-36cfcc67a0d7.mp4

## Demo: Speed Report
```bash
{
    "MaxSpeed": 12,
    "FastestVehicle": 1,
    "AverageVehicleSpeed": 8,
    "VehicleCount": 4
}
```
## Run your own video
Make sure your provide the file path of the input video to the other shell scripts in the end_to_end folder.
```bash
  sh /code/end_to_end/master_scipt.sh
```
## Checkout the slides 
https://docs.google.com/presentation/d/1IzZu4Pl6_Gr0gDDzN1gi-32mFEHhm8t7TMFzMLAyMyM/edit?usp=sharing