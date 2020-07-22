# Set up

This has been tested on AWS EC2 on a p3.2xlarge (V100) instance running the Deep Learning Base AMI (Ubuntu) Version 17.0 (ami-0ff00f007c727c376).

Build docker image.

    $ docker build -f docker/Dockerfile.gpu -t aminobert-gpu .

Run docker. This will launch a jupyter kernel. Note you will need port-forwarding enabled to 8888 if running on a remote machine.

    $ run_gpu_docker.sh

The relevant environment requirements can be gleaned by examining Dockerfile.gpu and requirements/requirements-py3.txt

Run the `Demo.ipynb` notebook. Cell 10 is where the magic happens, and is where the RGN would fit in.

**Note:** AminoBERT will output a (batch_size, max_seq_length, hidden_state_dim=768) matrix. For the CASP predictions we've done so far, for each sequence in the batch, I've clipped the corresponding (max_seq_length, 768) matrix to be (seq_length, 768) by removing the output vector corrsponding to the CLS token (index 0), and all other output vectors after the stop char of the sequence. Just FYI as it might influence how you feed AminoBERT's output to the RGN. 

