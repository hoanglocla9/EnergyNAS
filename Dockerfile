FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

MAINTAINER $(id -un)

COPY EnergyNAS/  /root/EnergyNAS

RUN apt-get update && apt-get install -y  python-is-python3 python3-pip

WORKDIR /root/EnergyNAS

RUN pip install tianshou pytorch-lightning colorama Cython scikit-learn typeguard==2.13 filelock==3.10 torchmetrics==0.11.4 json_tricks psutil requests astor \ 
				schema hyperopt pandas onnx onnxsim jsonlines onnxruntime tianshou==0.4.2 gym

RUN apt install npm -y

ENV strategy="random"
ENV trial_number=10
ENV n_gpus=1
ENV backbone_model="resnet"
ENV mode="debug"

EXPOSE 8081

#ENTRYPOINT ["python", "/root/EnergyNAS/run_nas.py"]
CMD ["sh", "-c", "python /root/EnergyNAS/run_nas.py --strategy $strategy --trial_number $trial_number --n_gpus $n_gpus --backbone_model $backbone_model --port 8081 --mode $mode"]
#CMD ["python", "/root/EnergyNAS/run_nas.py", "--strategy", "]
#ENTRYPOINT ["python run_nas.py"]
