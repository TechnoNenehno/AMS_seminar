# AMS IZZIV - final report
Bruno Černak      \
Light-weight Deformable Registration using Adversarial Learning with Distilling Knowledge\
https://github.com/TechnoNenehno/AMS_seminar.git      \
Main branch

## Method explanation


## Results
aggregated_results:    \
        LogJacDetStd        : 0.00001 +- 0.00000 | 30%: 0.00001   \
        TRE_kp              : 11.48576 +- 2.94629 | 30%: 12.24409 \
        TRE_lm              : 12.32114 +- 4.06665 | 30%: 12.93017 \
        DSC                 : 0.25145 +- 0.08298 | 30%: 0.20475   \
        HD95                : 49.13264 +- 13.05698 | 30%: 37.89221      \
brunoc@zigabpublic:~$   \

## Docker information
1. Git pull from the provided link above. (Main branch!)
2. Build Docker image. Training parameters can be adjusted in the entry.sh file before building.
3. Run the built image with: \
docker run -it --name container_name --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \ \
	      -v /media/FastDataMama/brunobanani/data/datasets/bizjak:/app/datasets \ \
	      -v weights_volume:/app/weights \    \
	      docker_image_name \
In the second line we bind the data into the container which is then used by train.py. The third line mounts a weights volume in which weights are stored after finishing training. From the obtained weights our deformations are built for us automatically by eval.py. Lastly our container stops and has deformations ready in the outputs folder. \
4. Copy deformations from the container to somewhere on disk with: \
docker cp container_name:/app/outputs/ /path/to/your/deformations \
5. Run evaluation with command below. Make sure you add your path to deformations + /outputs. \
docker run     --rm     -u $UID:$UID     \      \
	      -v /path/to/your/deformations/outputs:/input    \     \
 	      -v /path/to/your/output:/output/    \     \
 	      evaluation_image_name python evaluation.py -v   \

## Data preperation
Automatically implemented inside train.

## Train Commands
Defined in the entry.sh. Adjust as needed but keep in mind weights checkpoints are starting to get saved only after 1000 steps.

## Test Commands
Testing is done with step 5.
