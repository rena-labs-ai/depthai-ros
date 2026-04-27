#!/bin/bash
Help()
{
    echo "Build depthai_ros packages only (colcon --base-paths this repo)."
    echo "Skips depthai_filters_v3 (needs OpenCV contrib ximgproc)."
    echo "Run from your colcon workspace root (directory that contains src/)."
    echo
    echo "Build options:"
    echo "-s [1]   Set to 1 to build sequentially (longer, but saves RAM & CPU)"
    echo "-r [0]  Set to 1 to build in Debug mode. (RelWithDebInfo)"
    echo "-m [0]   Set to 1 for --merge-install (default is copy install, not symlink)."
    echo "-t [0]   Set to 1 to build tests."
    echo
}

sequential=1
release=0
merge=0
tests=0
build_type=Release
# empty = default colcon isolated install (copies, no --symlink-install)
install_type=
install_args=()
while getopts ":h:s:r:m:t:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      s) # Sequential executor
         sequential=$OPTARG;;
      r) # Build type
         release=$OPTARG;;
      m) # Install type
         merge=$OPTARG;;
      t) # Build tests
         tests=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

if [ "$release" == 0 ]
then
    build_type="RelWithDebInfo"
fi

if [ "$merge" == 1 ]; then
    install_type="merge-install"
fi
if [ -n "$install_type" ]; then
    install_args=(--"$install_type")
fi

build_testing_flag="OFF"
test_ros_driver_flag="OFF"
if [ "$tests" == 1 ]; then
  test_ros_driver_flag="ON"
  build_testing_flag="ON"
fi

DEPTHAI_ROS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WLS filter needs opencv2/ximgproc (contrib); skip unless OpenCV is built with contrib.
SKIP_DEPTHAI_FILTERS=(--packages-skip depthai_filters_v3)

install_label="${install_type:-isolated-copy}"
echo "Build type: $build_type, Install: $install_label, Testing: $tests"
echo "Packages path: $DEPTHAI_ROS_DIR"
if [ "$sequential" == 1 ]
then
    echo "Sequential build" && \
    MAKEFLAGS="-j1 -l1" colcon build \
        --base-paths "$DEPTHAI_ROS_DIR" \
        "${SKIP_DEPTHAI_FILTERS[@]}" \
        "${install_args[@]}" \
        --executor sequential \
        --cmake-args -DCMAKE_BUILD_TYPE=$build_type \
         -DBUILD_TESTING=$build_testing_flag \
         -DTEST_DEPTHAI_ROS_DRIVER=$test_ros_driver_flag \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DBUILD_SHARED_LIBS=ON
else
    echo "Parallel build" && \
    colcon build \
    --base-paths "$DEPTHAI_ROS_DIR" \
    "${SKIP_DEPTHAI_FILTERS[@]}" \
    "${install_args[@]}" \
    --cmake-args -DCMAKE_BUILD_TYPE=$build_type \
     -DBUILD_TESTING=$build_testing_flag \
     -DTEST_DEPTHAI_ROS_DRIVER=$test_ros_driver_flag \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
     -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
     -DBUILD_SHARED_LIBS=ON
fi
