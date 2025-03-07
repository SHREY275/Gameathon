# Fantasy Cricket Prediction API

This project is a **Flask-based API** that predicts the best fantasy cricket team using **machine learning (ML) and linear programming (LP)**. It dynamically selects players based on **predicted fantasy scores**, assigns a **captain & vice-captain**, and returns the result as a **CSV file**.

# 📂 Dataset
The datasets are stored in the `dataset/` directory, such as:
 `ausvssa.csv` (Australia vs South Africa data)
 `indvsban.csv` (India vs Bangladesh data)
 `pakvsnz.csv` (Pakistan vs New Zealand data)
 `indvspak.csv` (India vs Pakistan, merged dataset)

Each dataset contains columns like:
- `Player Name`
- `Role` (Batsman, Bowler, All-rounder, Wicketkeeper)
- `Team`
- `Fantasy_score`
- Match-related statistics

## ⚙️ Setup & Installation

### **2️ Install Dependencies**
If running locally (without Docker):
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Flask API Locally**
```sh
python app.py
```

API will be available at:
```
http://localhost:5000
http://127.0.0.1:5000
```

## 🐳 Running with Docker
### **1️⃣ Build Docker Image**
```sh
docker build -t shrey275/my_app .
```

### **2️⃣ Run the Docker Container**
```sh
docker run -p 5000:5000 -it shrey275/my_app
```

## 🏏 API Usage
### **1️⃣ Get Predicted Fantasy Team**
Use the following URL format:
```
http://127.0.0.1:5000/predict?dataset=indvspak&captain=Shubman%20Gill&vice_captain=Agha%20Salman
```

- `dataset=indvspak` → Specifies which dataset to use.
- `captain=Shubman Gill` → Sets the selected captain.
- `vice_captain=Agha Salman` → Sets the vice-captain.

### **2️⃣ Response**
- Returns a **CSV file** (`selected_team.csv`) with the best-selected team.
- If the captain or vice-captain is not found, the highest-ranked players are automatically assigned.


### **2️⃣ Push to Docker Hub**
```sh
docker push shrey275/my_app
```

### **3️⃣ Run Updated Container**
```sh
docker run -p 5000:5000 -it shrey275/my_app
```

### **4️⃣ Pull Updated Image on Another Machine**
```sh
docker pull shrey275/my_app:latest
docker run -p 5000:5000 -it shrey275/my_app
```

## 🛑 Stopping & Removing Containers
Check running containers:
```sh
docker ps
```
Stop a running container:
```sh
docker stop <container_id>
```
Remove a container:
```sh
docker rm <container_id>
```
Remove an image:
```sh
docker rmi yourusername/my_app
```

## 💡 Troubleshooting
- If port `5000` is in use:
  ```sh
  netstat -ano | findstr :5000  # Check the process using the port
  taskkill /PID <pid> /F        # Kill the process
  ```
- If `ModuleNotFoundError: No module named 'numpy'` occurs, rebuild the image using:
  ```sh
docker build --no-cache -t yourusername/my_app .
  ```

## 📜 License
This project is open-source under the **MIT License**.

---

🔥 **Now, you're ready to predict the best fantasy cricket team!** 🏏🚀

