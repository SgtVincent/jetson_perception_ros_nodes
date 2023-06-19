#!/usr/bin/env bash

CONTAINER_IMAGE="sgtvincent/ros:noetic-raft-stereo"
PROJECT_ROOT="/ros_deep_learning"

DOCKER_ARGS="-e ROS_MASTER_URI=http://192.168.55.100:11311 -e ROS_IP=192.168.55.1"

USER_VOLUME="\
--volume /home/jetson/repo/jetson-inference/data:/jetson-inference/data \
--volume /home/jetson/repo/ros_deep_learning:$PROJECT_ROOT/src/ros_deep_learning \
--volume /home/jetson/repo:/repo"

USER_COMMAND="" 

# check for V4L2 devices
V4L2_DEVICES=" "

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

# # check for display
# DISPLAY_DEVICE=" "

# if [ -n "$DISPLAY" ]; then
# 	# give docker root user X11 permissions
# 	sudo xhost +si:localuser:root
	
# 	# enable SSH X11 forwarding inside container (https://stackoverflow.com/q/48235040)
# 	XAUTH=/tmp/.docker.xauth
# 	xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
# 	chmod 777 $XAUTH

# 	DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
# fi

echo "CONTAINER_IMAGE: $CONTAINER_IMAGE"
echo "DOCKER_ARGS:      $DOCKER_ARGS"
echo "USER_VOLUME:     $USER_VOLUME"
echo "USER_COMMAND:    '$USER_COMMAND'"
echo "DISPLAY_DEVICE:  $DISPLAY_DEVICE"

# keep container running in background and DO NOT remove for devs purpose
sudo docker run --runtime nvidia -itd --network host \
    $DOCKER_ARGS \
	$DISPLAY_DEVICE $V4L2_DEVICES \
	$USER_VOLUME $CONTAINER_IMAGE $USER_COMMAND