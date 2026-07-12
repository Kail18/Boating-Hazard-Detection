# AI Usage Log

## Prompt 1 | 8:25am 07/11/2026

1. Alright lets start setting up the project. The purpose of this chat is to get us caught up on the midterm project point.

2. It provided an entire repository set up. This was not intended. I usually start chats or projects describing the main purpose of the chat so that it learns the rules and regulations that I set on the chat. I guess it took me saying start setting up the project literally. I have uploaded the repository set up and I will start going through each of the files.

## Prompt 2 | 9:00am 07/11/2026

1. Finalize and import the maritime dataset.

2. It provided a script test_lars_to_yolo.py. It also provided scripts to bash command for running different tests on the dataset that I downloaded from LARS.

## Prompt 3 | 10:34 07/11/2026

1. Give me a markdown of what we have done and what we will do next for me to add to the midterm progress in the README.md

2. It gave me an extensive markdown.

## Prompt 4 | 10:35 07/11/2026

1. give me a condensed summary markdown with key points to add. I do not need a breakdown of each step

2. The condensed summary was added to the README.md file.

## Prompt 5 | 10:49 07/11/2026

1.  Visualize sample bounding boxes to confirm annotation accuracy

Lets start working on this step. This time do not just complete the step entirely. Lets break this step down into smaller steps and explain each process along the way. Also when adding methods or functions make sure to use more documentation give more detail to the """ Documentation """ to better explain the purpose of those methods and functions.

2. It gave me a python script with much better documentation for functions and methods and an in depth break down on what we are doing in this step.

3. **Notes** I added this prompt because Prompt 1 and Prompt 2 had extensive amount of work completed with very little prompting. This was not what I wanted, because the purpose of this project is for me to learn the process. When it was pushing out that many steps at once it was hard for me to keep up with the scripts. This prompt should reallign the AI to make it easier to learn from rather than just completing the project.

## Prompt 6 | 11:04 07/11/2026

1. Training images: 2605
   Validation images: 198

Configured classes:
0: vessel
1: buoy
2: swimmer
3: paddle_board
4: floating_obstacle

Stage 1 verification completed successfully.

output after running script

2. It next had me add the shared image-extension constants, Add a function for locating the label directory, Add a function that finds one matching pair, and cleaned up the main() inside KailMcGuire-CS898BA-Project/scripts/visualize_yolo_labels.py

## Prompt 7 | 11:46 07/11/2026

1. Training images: 2605
   Validation images: 198

Configured classes:
0: vessel
1: buoy
2: swimmer
3: paddle_board
4: floating_obstacle

Stage 1 dataset verification completed successfully.

Sample annotated image:
Image: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/images/train/davimar_seq_01_00017.jpg
Label: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/labels/train/davimar_seq_01_00017.txt
Objects in label file: 3

First annotation lines:
0 0.834013 0.494767 0.331975 0.437198
4 0.331975 0.517311 0.003623 0.004026
0 0.569520 0.478261 0.044837 0.024155

Stage 2 image-label pairing completed successfully.

2. Stage 3: Parse annotations and calculate pixel coordinatew:

   - Add the new imports
   - Add a YoloAnnotation
   - Add the annotation parsing function
   - Add a function for reading image diminsions
   - Add the normalized-to-pixel conversion function
   - update main()

   ### What Stage 3 verifies:

   When this succeeds, we will have confirmed that:

   - Every row contains five valid values.
   - Class IDs match the configured class names.
   - Normalized coordinates are within valid ranges.
   - The image can be read by OpenCV.
   - The image dimensions are available.
   - YOLO coordinates convert into valid pixel boxes.
   - Small objects remain visible as very small pixel regions.

   The next stage will use these pixel corner coordinates with cv.rectangle() and cv.putText() to draw labels on a copy of the image.

## Prompt 8 | 12:01 07/11/2026

1. Training images: 2605
   Validation images: 198

Configured classes:
0: vessel
1: buoy
2: swimmer
3: paddle_board
4: floating_obstacle

Stage 1 dataset verification completed successfully.

Sample annotated image:
Image: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/images/train/davimar_seq_01_00017.jpg
Label: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/labels/train/davimar_seq_01_00017.txt
Objects in label file: 3

Image dimensions:
Width: 2208 pixels
Height: 1242 pixels

Parsed annotations and pixel coordinates:

Object 1
Class: 0 (vessel)
Normalized box: center=(0.834013, 0.494767), size=(0.331975, 0.437198)
Pixel corners: left=1475, top=343, right=2207, bottom=886
Pixel size: 732 × 543

Stage 3 coordinate conversion completed successfully.

Stage 2 image-label pairing completed successfully.

Object 2
Class: 4 (floating_obstacle)
Normalized box: center=(0.331975, 0.517311), size=(0.003623, 0.004026)
Pixel corners: left=729, top=640, right=737, bottom=645
Pixel size: 8 × 5

Stage 3 coordinate conversion completed successfully.

Stage 2 image-label pairing completed successfully.

Object 3
Class: 0 (vessel)
Normalized box: center=(0.569520, 0.478261), size=(0.044837, 0.024155)
Pixel corners: left=1208, top=579, right=1307, bottom=609
Pixel size: 99 × 30

Stage 3 coordinate conversion completed successfully.

Stage 2 image-label pairing completed successfully.

This is the output after implementing stage 3.

Before we move onto the next stage lets create a directory for the visualization_yolo lets take out the functions from each stage have them in their own class and file and then import that class into the visualization_yolo file. This should help clean up the clutter in this file

2. This was multiple steps I added a couple of different refactoring options to better clean up the visualizaiton directory.

## Prompt 9 | 12:17 07/11/2026

1. YOLO Dataset Verification

---

Configuration: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/dataset.yaml
Dataset root: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo
Training images: 2605
Validation images: 198

Configured classes:
0: vessel
1: buoy
2: swimmer
3: paddle_board
4: floating_obstacle

Stage 1 dataset verification completed successfully.

Sample annotated image:
Image: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/images/train/davimar_seq_01_00017.jpg
Label: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/labels/train/davimar_seq_01_00017.txt
Objects in label file: 3

Stage 2 image-label pairing completed successfully.

Image dimensions:
Width: 2208 pixels
Height: 1242 pixels

Parsed annotations and pixel coordinates:

Object 1
Class: 0 (vessel)
Normalized box: center=(0.834013, 0.494767), size=(0.331975, 0.437198)
Pixel corners: left=1475, top=343, right=2207, bottom=886
Pixel size: 732 × 543

Object 2
Class: 4 (floating_obstacle)
Normalized box: center=(0.331975, 0.517311), size=(0.003623, 0.004026)
Pixel corners: left=729, top=640, right=737, bottom=645
Pixel size: 8 × 5

Object 3
Class: 0 (vessel)
Normalized box: center=(0.569520, 0.478261), size=(0.044837, 0.024155)
Pixel corners: left=1208, top=579, right=1307, bottom=609
Pixel size: 99 × 30

Stage 3 coordinate conversion completed successfully.

This is the output after refactoring

2. Stage 4

   - added the new py file to visualization_yolo annotation_renderer
   - updated the visualization_yolo init.py
   - updated the visualize_yolo script import and main()

## Prompt 10 | 14:31 07/11/2026

1. Training images: 2605
   Validation images: 198

Configured classes:
0: vessel
1: buoy
2: swimmer
3: paddle_board
4: floating_obstacle

Stage 1 dataset verification completed successfully.

Sample annotated image:
Image: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/images/train/davimar_seq_01_00017.jpg
Label: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/labels/train/davimar_seq_01_00017.txt
Objects in label file: 3

Stage 2 image-label pairing completed successfully.

Image dimensions:
Width: 2208 pixels
Height: 1242 pixels

Parsed annotations and pixel coordinates:

Object 1
Class: 0 (vessel)
Normalized box: center=(0.834013, 0.494767), size=(0.331975, 0.437198)
Pixel corners: left=1475, top=343, right=2207, bottom=886
Pixel size: 732 × 543

Object 2
Class: 4 (floating_obstacle)
Normalized box: center=(0.331975, 0.517311), size=(0.003623, 0.004026)
Pixel corners: left=729, top=640, right=737, bottom=645
Pixel size: 8 × 5

Object 3
Class: 0 (vessel)
Normalized box: center=(0.569520, 0.478261), size=(0.044837, 0.024155)
Pixel corners: left=1208, top=579, right=1307, bottom=609
Pixel size: 99 × 30

Stage 3 coordinate conversion completed successfully.

Rendered annotation visualization:
Source image: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/data/processed/lars_yolo/images/train/davimar_seq_01_00017.jpg
Saved image: /Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/FinalProject/KailMcGuire-CS898BA-Project/outputs/figures/label_verification/davimar_seq_01_00017_annotated.jpg
Boxes drawn: 3
Image size: 2208 × 1242

Stage 4 bounding-box rendering completed successfully.

here is the output of Stage 4.

## Prompt 11 | 15:36 07/012/2026

1. we also might want to consider doing a YOLO and Fast R-CNN since the YOLO research paper discussed seeing the most promising results with a combination of the two

Note: I prompted this since the current trajectory was to just use the YOLO and R-CNN seperate. I noticed the best results were seen with a combination of the two in the YOLO.pdf paper.

2. Chat GPT is now alligned with the direction that I believe is a better fit for the project. It broke down the outputs in stages.
