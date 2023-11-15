POSITIONAL=()
export debug=0
# $#是变量的个数
# -gt是大于
# 将所有的参数过一遍，输入参数是--numworkers 2 --batchsize 1024这样的形式
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -d|--debug)
      export debug=1
      shift # past argument
      ;;
    -t|--test)
      export test=true
      shift # past value
      ;;
    -g|--gres)
      export gpus="$2"
      shift # past value
      shift # past value
      ;;
    --lr)
      export lr="$2"
      shift # past value
      shift # past value
      ;;
    -b|--batch_size)
      export batch_size="$2"
      shift # past value
      shift # past value
      ;;
    -j|--num_workers)
      export num_workers="$2"
      shift # past value
      shift # past value
      ;;
    -s|--shell)
      export script="$2"
      shift # past value
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

echo POSITIONAL ARGS:
for i in ${POSITIONAL[@]}; do
  echo $i
done

# hostname=$(cat /proc/sys/kernel/hostname)
# if [ "$hostname" = "node06" ]; then
#   echo export cluster=node06
#   export cluster=node06
# elif [ "$hostname" = "master" ]; then
#   echo export cluster=master
#   export cluster=master
# elif [ "$hostname" = "node09" ]; then
#   echo export cluster=node09
#   export cluster=node09
# elif [ "$hostname" = "ubuntu" ]; then
#   echo cuda=2 to 9
#   export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7,8,9"
# fi


export log_file_path="$HOME/$RANDOM.txt"

run() {
  echo ---------------------exp_args---------------------
  echo "${exp_args[@]}"
  echo ---------------------run_args---------------------
  echo "${run_args[@]}"
  echo ---------------------extra_args---------------------
  echo "${POSITIONAL[@]}"
  bash base.sh "${exp_args[@]}" "${run_args[@]}" "${POSITIONAL[@]}"

  if test "$debug" = false; then
    if [ $? -eq 0 ]; then
      curl "https://api.day.app/AWhYUgjkHvmi9Hn8GNnkta/Finished/$run_name($exp_name)"
    else
      curl "https://api.day.app/AWhYUgjkHvmi9Hn8GNnkta/Failed/$run_name($exp_name)"
    fi
  fi
}