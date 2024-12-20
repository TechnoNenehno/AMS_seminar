# AMS IZZIV - final report
Bruno Černak      \
Light-weight Deformable Registration using Adversarial Learning with Distilling Knowledge\
https://github.com/TechnoNenehno/AMS_seminar.git (main branch)

## Method explanation
In this document, a new algorithm is presented for training lighter, less computationally demanding neural networks. The algorithm is based on supervised learning, where a "teacher" neural network, trained on large datasets and computationally expensive, transfers its knowledge and key information to an untrained "student" neural network. This approach enables faster learning for the student neural network while reducing the number of parameters. The new learning method doesn't rely on competition between networks, rather the student neural network uses the knowledge transferred from the teacher network as its foundational truth. The teacher neural network provides well-defined image deformations, differing from traditional algorithms that typically optimize nonlinear spatial similarities.

The task of the student network is to create a prediction function F that takes a moving image and a fixed image as input. The goal is to predict a deformation through a cascade framework(recursive cascaded networks) aligning the inputs with convolution and deconvolution. The network also calculates and minimizes a composite loss function which consists of reconstruction loss and discrimination loss. The losses are considered on every cascade and progressivly refined.

The teacher transfers distiled deformations to the student, upon which the student learns. For this part we did not have the time to train the computationaly expensive tacher network on our data nor did the paper describe how to prepare the Teacher deformations dataset. In my opinion this paper was not well suited for the task of this challenge.

## Results
aggregated_results:    \
        LogJacDetStd        : 0.00001 +- 0.00000 | 30%: 0.00001   \
        TRE_kp              : 11.48576 +- 2.94629 | 30%: 12.24409 \
        TRE_lm              : 12.32114 +- 4.06665 | 30%: 12.93017 \
        DSC                 : 0.25145 +- 0.08298 | 30%: 0.20475   \
        HD95                : 49.13264 +- 13.05698 | 30%: 37.89221      \
brunoc@zigabpublic:~$   

## Docker information
1. Git pull from the provided link above. (main branch!)
2. Build Docker image. Training parameters can be adjusted in the entry.sh file before building.
3. Run the built image with: \
docker run -it --name container_name --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \ \
	      -v /media/FastDataMama/brunobanani/data/datasets/bizjak:/app/datasets \ \
	      -v weights_volume:/app/weights \    \
	      docker_image_name 

In the second line we bind the data into the container which is then used by train.py. The third line mounts a weights volume in which weights are stored after finishing training. From the obtained weights our deformations are built for us automatically by eval.py. Lastly our container stops and has deformations ready in the outputs folder.

4. Copy deformations from the container to somewhere on disk with: \
docker cp container_name:/app/outputs/ /path/to/your/deformations 

5. Run test/evaluation with command below. Make sure you add /path/to/your/deformations and paste /outputs. \
docker run     --rm     -u $UID:$UID     \      \
	      -v /path/to/your/deformations/outputs:/input    \     \
 	      -v /path/to/your/output:/output/    \     \
 	      evaluation_image_name python evaluation.py -v   

## Data preperation
Data is decimated inside the train.py function to shape (128,128,128) before processing.

## Train Commands
Defined in the entry.sh. Adjust as needed but keep in mind weights checkpoints are stored after 1000 steps.

## Test Commands
Testing is done with step 5.

## Example run
docker build -t seminar_test .      

docker run -it --name treniranje --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \     \
	-v /media/FastDataMama/brunobanani/data/datasets/bizjak:/app/datasets \ \
	-v weights_volume:/app/weights \    \
	seminar_test      

docker cp testiranje:/app/outputs/ /home/brunoc/data/deformacije  

docker run     --rm     -u $UID:$UID     \      \
	-v /home/brunoc/data/deformacije/outputs:/input    \  \
 	-v /home/brunoc/data/output:/output/    \ \
 	gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration \     \
      python evaluation.py -v

