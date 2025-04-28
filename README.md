## Real-Time FaceMask-Detection
This project detects whether a person is wearing a face mask or not using a live webcam feed, and displays the result in a web interface built with Flask. It uses deep learning (CNN model) for classification and OpenCV Haar Cascade for real-time face detection

# Dataset 
The data used can be downloaded through this [link](https://data-flair.training/blogs/download-face-mask-data/)
The dataset consists of 1376 images with 690 images containing images of people wearing masks and 686 images with people without masks.
It is an excellent dataset for people who want to try learning techniques of deep learning for face mask detection




## 📂 Project Folder Structure


```
Real-Time-Face-Mask-Detection/
├── static/
│   └── styles.css                 # CSS file for styling the web interface
├── templates/
│   └── index.html                  # Main HTML template for the application
├── App.py                          # Flask application to serve the interface
├── facemask.py                     # Script for real-time face mask detection
├── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
├── live.py                         # Additional real-time detection script
├── mymodel.h5                      # Trained CNN model for mask detection
├── requirements.txt                # Python libraries to install
└── README.md                       # Project documentation
```





## How to Use

To use this project on your system, follow these steps:

1.Clone this repository onto your system by typing the following command on your Command Prompt:

```
git clone https://github.com/Karan-Malik/FaceMaskDetector.git
```
followed by:

```
cd FaceMaskDetector
```

2. Download all libaries using::
```
pip install -r requirements.txt
```

3. Run facemask.py by typing the following command on your Command Prompt:
```
python facemask.py
```

#### The Project is now ready to use !!
