To create the Yolo dataset directory, follow the next steps:

1. Create inside the Week3 folder the following directory:

        Week3 ---> dataset ---> annotations ---> test
                         |                |
                         |                |---> train
                         |
                         |---> frames ---> test
                                    |
                                    |---> train
                                     

2. Move the gt file inside the dataset folder:

3. Move the AICity_data folder inside the Week3. The directory should look like this.

        Week3 ---> dataset ---> annotations ---> test
             |            |                |
             |            |                |---> train
             |            |
             |            |---> frames ---> test
             |            |           |
             |            |           |---> train
             |            |
             |            |---> <gt file> ai_challenge_s03_c010-full_annotation.xml
             |
             |---> AICity_data

4. Once the directory is created, run the following command to prepare the annotations.
    python create_yolo_annotation.py dataset/ai_challenge_s03_c010-full_annotation.xml --train_output dataset/annotations/train --test_output dataset/annotations/test

5. run the following command to prepare the frames.
    python create_yolo_dataset.py AICity_data/train/S03/c010/vdo.avi