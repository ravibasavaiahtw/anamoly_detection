## Setting up environment
1. Creating conda environmnent:
```
conda create -n "hack-autoencoder" python=3.9
conda activate "hack-autoencoder"
pip install -r requirements.txt
```
2. activating conda environment on VSCode or activating via terminal:
```
conda activate hack-autoencoder
```
3. Execute Anomaly_Detection_Autoencoder.ipynb
4. Initiate API:
```
cd hackathon_anomaly_detection/notebook
flask --app inference_api run --port 5001
```