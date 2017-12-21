# Download the necessary train and test data, as well as a pre-trained model
wget -O data.tgz https://drive.switch.ch/index.php/s/ij4XoRAJtGjBA8m/download && \
  tar -xvzf data.tgz
# Run the main script that generates our best predictions
python run.py
