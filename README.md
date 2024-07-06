# COVID-19, Pneumonia, and Normal Classification from X-ray Images

Accurate diagnosis of respiratory conditions is crucial for effective treatment and patient care. This project leverages deep learning techniques to classify chest X-ray images into three categories:

- COVID-19
- Pneumonia
- Normal

The model combines the powerful feature extraction capabilities of ResNet50 with the sequence processing strengths of Gated Recurrent Units (GRUs). This hybrid architecture ensures high accuracy in distinguishing between different types of lung conditions.

To make this tool accessible to healthcare professionals and researchers, I have developed a user-friendly web application using [Streamlit](https://streamlit.io/). This application allows users to upload X-ray images and receive instant classification results, aiding in quick and informed decision-making.

## Screenshot

![Web App Screenshot](https://github.com/nawaz0x1/X-Ray-Classification/blob/master/Screenshot/Screenshot%201.png)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/nawaz0x1/X-Ray-Classification.git
    cd X-Ray-Classification
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the `model.keras` file from the [GDrive](https://drive.google.com/file/d/1OIo7oakSxPU3K51Cqnz6Dk1hsF16JuOu) link and replace it with `Model/model.keras`.

## Usage

To run the Streamlit web app:

```bash
streamlit run App.py
```

## Acknowledgement

The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset).

