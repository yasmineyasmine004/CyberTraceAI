# CyberTrace AI

AI-assisted network intrusion detection and security incident analysis using CICIDS2017 network-flow data.

## Participant
- Participant name: ZERIGUI Alia Yasmine

## Problem Statement
Traditional security systems can struggle with previously unseen attacks because they often depend on known signatures or fixed rules. Abnormal behaviour may appear as subtle changes across users, devices, processes or network activity and may only become meaningful when several events are considered together.

## Problem Description
Network traffic can contain malicious activities that are difficult to identify through manual inspection. CyberTrace AI explores machine-learning-based anomaly detection and specialized PortScan classification to help analysts investigate suspicious network flows.

## Proposed Solution
CyberTrace AI combines an Isolation Forest model for general anomaly detection with a Random Forest classifier specialized in distinguishing Benign and PortScan traffic. A Streamlit dashboard presents analysis results and incident-related information.

## Key Features
- Interactive Streamlit dashboard.
- General anomaly detection using Isolation Forest.
- Specialized PortScan classification using Random Forest.
- Incident grouping and correlation heuristics.
- Visual summaries and suspicious-flow analysis.
- CSV analysis through the dashboard, subject to the expected feature format.

## Technology Stack
- Python
- Pandas and NumPy
- scikit-learn
- Streamlit
- Plotly
- Joblib
- Parquet via PyArrow

## Methodology
1. Load and inspect network-flow data.
2. Prepare numeric features for machine-learning models.
3. Train Isolation Forest using benign traffic.
4. Train a Random Forest classifier for Benign versus PortScan traffic.
5. Analyze suspicious flows and display results in the dashboard.
6. Use incident-correlation heuristics to help organize findings.

## Architecture / Workflow
CICIDS2017 flow data -> feature preparation -> machine-learning models -> detection results -> incident analysis -> Streamlit dashboard.

## Installation
Use Python 3.10 or a compatible Python version supported by the installed dependencies.

1. Clone this repository:
   `git clone [GITHUB_REPOSITORY_URL]`
2. Enter the project directory:
   `cd CyberTraceAI`
3. Create and activate a virtual environment:
   `python -m venv .venv`
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Obtain CICIDS2017 data and place the required Parquet files in `data/raw/`, following the filenames expected by the training scripts.
6. Train and save the model bundles:
   `python train_model_bundle.py`
7. Start the dashboard:
   `streamlit run app.py`

## Usage
Open the local URL displayed by Streamlit. Use the dashboard to explore detection results and, where available, upload a compatible CSV file for model analysis.

The model bundles are generated locally and are not included in this repository by default. Run the training script before using dashboard features that require these bundles.

## Dataset Information
This project uses flow data derived from CICIDS2017. Dataset files are not included in this repository. Obtain the dataset from the Canadian Institute for Cybersecurity and comply with its terms of use and attribution requirements.

Dataset source: https://www.unb.ca/cic/datasets/ids-2017.html

The supplied flow files used in this project contain numeric network-flow features and a label column. The project does not reconstruct packet-level traffic from these files.

## Models Used
- **Isolation Forest:** unsupervised anomaly detection trained on benign traffic.
- **Random Forest:** supervised classifier specialized in Benign versus PortScan traffic.

The Random Forest's reported internal validation used a random split from the same CICIDS2017 source file. Its near-perfect score is not proof of generalization to independent datasets or real-world traffic.

## Results
The internal Random Forest validation produced the following confusion matrix, with rows and columns ordered as Benign and PortScan:

`[[29392, 0], [1, 488]]`

This corresponds to one missed PortScan in the reported validation split. These results are preliminary and should not be interpreted as independently verified real-world performance. Isolation Forest performance depends on the selected threshold and traffic distribution.

## Screenshots and Demo
- Dashboard screenshots: [ADD SCREENSHOTS TO THE REPOSITORY]
- Demo video or link: [ADD LINK IF AVAILABLE]

## Limitations
- Validation is based on CICIDS2017 and does not establish performance on independent real-world traffic.
- PortScan classification is specialized; a prediction of non-PortScan does not mean traffic is benign.
- The available flow files do not provide all packet-level details, IP addresses, or timestamps needed for full forensic reconstruction.
- Some incident timelines or correlations may be synthetic or heuristic and must be clearly identified as such.
- Model thresholds and results may vary with the dataset and configuration.

## Future Scope
- Evaluate on independent datasets and real network captures.
- Improve generalization to new attack types.
- Calibrate thresholds using a separate validation dataset.
- Improve incident correlation and explainability.
- Add reproducible evaluation reports.

## Credits and Attributions
- CICIDS2017 dataset: Canadian Institute for Cybersecurity, University of New Brunswick.
- Open-source libraries: Python, Pandas, NumPy, scikit-learn, Streamlit, Plotly, PyArrow and Joblib.
- AI-assisted development tools were used during development. Specific disclosure should follow the hackathon rules.

## Security
Do not commit passwords, API keys, access tokens, private credentials, raw datasets, or generated model files.
