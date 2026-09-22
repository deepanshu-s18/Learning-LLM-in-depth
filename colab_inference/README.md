# 🛠️ Colab + 5TB Google Drive Inference & Model Hub

A production-style system to download, store, and serve open-source LLMs and your custom from-scratch models using Google Colab and Google Drive.

---

## 📁 Files Created

1. **[1_Drive_Model_Hub_Downloader.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/colab_inference/1_Drive_Model_Hub_Downloader.ipynb)**:
   - Mounts your 5TB Google Drive at `/content/drive/MyDrive/LLM_Model_Hub`.
   - Has one-click downloads for top quantized models: Llama-3-8B-AWQ, Qwen-2.5-Coder-7B-AWQ, DeepSeek-R1-Distill-7B-AWQ, Mistral-7B-AWQ, and custom HuggingFace repos.
   - Includes `ScratchModelManager` class to save and load your custom from-scratch models (weights, configs, checkpoints) directly to/from Drive.
   - Includes a storage inspector to track model sizes on Drive.

2. **[drive_model_manager.py](file:///Users/shobhitagnihotri/Desktop/internship/colab_inference/drive_model_manager.py)**:
   - Standalone Python module you can import into your training scripts or notebooks:
     ```python
     from drive_model_manager import GoogleDriveModelHub
     hub = GoogleDriveModelHub()
     hub.save_scratch_model(my_model, "deepseek_scratch_v1", config=cfg)
     ```

3. **[2_vLLM_Inference_Server.ipynb](file:///Users/shobhitagnihotri/Desktop/internship/colab_inference/2_vLLM_Inference_Server.ipynb)**:
   - Starts a high-throughput `vLLM` server on Colab using models stored in your Drive.
   - Connects `pyngrok` to expose an OpenAI-compatible REST API endpoint (`https://xyz.ngrok.app/v1`).

4. **[client_test.py](file:///Users/shobhitagnihotri/Desktop/internship/colab_inference/client_test.py)**:
   - Run this locally on your Mac in Antigravity to send streaming requests to your Colab server.
   - Automatically computes **Time to First Token (TTFT)** and **tokens/sec** metrics.

---

## 🚀 Step-by-Step Workflow

### Step 1: Download Models to Drive
1. Open Google Colab and upload `1_Drive_Model_Hub_Downloader.ipynb`.
2. Run the cells to download your desired model (e.g. `Llama-3-8B-Instruct-AWQ`). It will be saved permanently in your Google Drive.

### Step 2: Start vLLM Server
1. Upload `2_vLLM_Inference_Server.ipynb` to Colab with GPU enabled.
2. Put your free ngrok token in cell 3.
3. Run all cells. It will output a public API URL (e.g. `https://xxxx.ngrok-free.app/v1`).

### Step 3: Call your API from Mac / Antigravity
1. Open `client_test.py` on your Mac.
2. Paste the ngrok URL into `NGROK_BASE_URL`.
3. Run `python client_test.py` in your terminal to chat and measure live tokens/second.
