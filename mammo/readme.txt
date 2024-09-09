Project Overview

This project processes audio files (.m4a) and converts them into textual reports using a machine learning pipeline. The folder contains the necessary code and dependencies for running the project inside a Docker container.

File Descriptions

	•app.py: The main application script that processes the audio files, transcribes them into text, and generates the corresponding reports. It is designed to run inside a Docker container and integrates with the 		necessary APIs and models.

	•commands: Contains commands for creating and running the docker image

	•Dockerfile: A configuration file that sets up the environment for the project. It specifies the base image and instructions to install all dependencies required for the application.

	•requirements.txt: A list of Python libraries that are needed to run the application. These include libraries for web frameworks, machine learning, audio processing, and report generation (e.g., streamlit, 			pandas, openai, pydub, etc.) 

	•pt_48.m4a, pt_49.m4a, pt_50.m4a: Sample audio files that will be processed by the application. The .m4a files are the input audio that will be converted into text and analyzed to produce the final report.

Instructions for Running the Project

	1.Docker Setup:
	•Ensure that Docker is installed on your machine.
	•Build the Docker image by running commands in the command file.

	2.Python Environment:
	•If running locally, install the dependencies using the following command:
	 pip install -r requirements.txt
	•run the app.py using:
	streamlit run app.py

	3.Processing Audio Files:
	•The application will take the .m4a audio files as input and generate a transcribed text report.
	•You can modify or replace the existing audio files (pt_48.m4a, pt_49.m4a, and pt_50.m4a) with your own inputs.

Dependencies:

Refer to the requirements.txt file for the complete list of required packages.

Contact

For any issues or questions, please reach out me at ce1231156@iitd.ac.in or 8178056230
