# Monocular Depth Estimation 

## Summary 

Overall deployed models summary: 
| Model                      | Speed   | Quality on custom data | NYU Depth-v2 RMSE | Compatible Docker Base Image           |
|----------------------------|---------|------------------------|-------------------|----------------------------------------|
| FastDepth (2019 ICRA)      | ~60 FPS | Terrible               | 0.604             | dustynv/ros:noetic-pytorch-l4t-r35.3.1 |
| GuidedDecoding (2022 ICRA) | ~6 FPS  | Good                   | 0.478             | dustynv/ros:noetic-pytorch-l4t-r35.3.1 |
| GLPN (SOTA in 2022)        | <1 FPS  | Perfect                | 0.344             | dustynv/ros:noetic-pytorch-l4t-r35.3.1 |

Qualitative result on our custom data: 
<!-- a table for images -->
| Input Image | FastDepth | GuidedDecoding | GLPN |
|-------------|-----------|----------------|------|


## Fast Depth 



## GuidedDecoding 



### Extra dependencies: 
```bash
pip3 install matplotlib
```

### Preparing pre-trained models 
You should place the pre-trained feature extractor [DDRNet-23 slim](https://drive.google.com/file/d/1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4/view). Then place it under folder[./scripts/monocular_depth_estimation/GuidedDecoding/model/weights](./scripts/monocular_depth_estimation/GuidedDecoding/model/weights) or change the hard-coded path in (./scripts/monocular_depth_estimation/GuidedDecoding/model/DDRNet_23_slim.py)[./scripts/monocular_depth_estimation/GuidedDecoding/model/DDRNet_23_slim.py] on line 361.

Then download the pre-trained model weights (here)[https://drive.google.com/file/d/1TNTUUve5LHEv6ERN6v9aX2eYw1-a-4bO/view?usp=sharing]. You should specify the path in config file (./config/default.yaml)[./config/default.yaml].

Please refer to [official repo](https://github.com/mic-rud/guideddecoding) for detailed instructions and trouble shooting.

## GLPN