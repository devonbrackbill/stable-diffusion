
apt update
apt-get install unzip
# setup AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

git clone https://github.com/devonbrackbill/stable-diffusion.git -y
cd stable-diffusion
pip install -r requirements.txt
pip install -r requirements2.txt
pip install --upgrade keras
pip uninstall -y torchtext
pip install huggingface_hub
pip install cairosvg pandas

# cairosvg dependencies
apt install -y libcairo2-dev
# cv2 dependencies
apt-get install ffmpeg libsm6 libxext6  -y