# 📚 Distraction Analysis in Online Learning Environments

A real-time system to detect and analyze student distraction during online classes using computer vision and behavioral analytics.

## 📝 Description
The rise of online education faces a significant challenge in maintaining student engagement and focus due to numerous distractions and the absence of real-time monitoring. The **Distraction Analysis** project offers a robust framework to address this issue by measuring and enhancing learner focus in virtual environments.

The system leverages a combination of behavioral data, machine learning, and real-time feedback to create a standardized metric for evaluating distraction levels during online sessions. It analyzes a learner's head pose, gaze direction, drowsiness, and facial expressions to provide a dynamic score that reflects their level of engagement. This approach provides educators with actionable insights to refine course delivery and helps learners develop self-awareness for improved academic outcomes.

## ⚙️ Requirements
This project requires Python and Conda. All dependencies are managed via the `environment.yml` file.

## 📦 Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Distraction_Analysis
    ```
3.  **Create the Conda environment from the `environment.yml` file:**
    ```bash
    conda env create -f environment.yml
    ```
4.  **Activate the new environment:**
    ```bash
    conda activate myproject_env
    ```

## ▶️ Usage
To run the application, ensure your webcam is connected and active.

1.  **Run the main script from your terminal:**
    ```bash
    python main.py
    ```
2.  **Using the Application:**
    -   A window will open, displaying your webcam feed with real-time analysis.
    -   The dashboard will show your concentration percentage, drowsiness status, gaze direction, and head orientation.
    -   To exit the application, press the `Esc` key.
    -   A graph of your concentration over time will be displayed after the application closes.
