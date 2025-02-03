# 🚀 Setting Up the Project

## 🛠️ Clone the Repository  
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## 🌍 Activate the Virtual Environment

Widows:
```sh
venv\Scripts\Activate
```

```sh
source venv/bin/activate
```

## 🌍 Deactivate

```sh
deactivate
```

## 📦 Install Dependencies

```sh
pip install -r requirements.txt
```
## 🔄 Update Dependencies Before Pushing

```sh
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push origin main
```