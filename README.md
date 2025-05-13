# Iris Osztályozás MLOps Projekt

Ez a projekt bemutat egy teljes MLOps (Machine Learning Operations – gépi tanulási műveletek) folyamatot egy gépi tanulási modellre, amely íriszvirágokat osztályoz a mért jellemzőik alapján. A projekt célja a modern gépi tanulási eszközök és gyakorlatok integrálása egy éles környezetben.

## Projekt Áttekintés

Az Iris osztályozó rendszer a scikit-learn könyvtárat használja a klasszikus Iris adathalmazon történő modelltanításhoz. A projekt egy teljes MLOps-folyamatot valósít meg, amely tartalmazza:
- Modell tanítását és verziókövetését
- API telepítést
- Felhasználóbarát irányítópultot
- Modell monitorozást és riportolást
- Automatizált munkafolyamat ütemezést

## Projekt Elemei

### FastAPI Backend
- REST API előrejelzésekhez
- Valós idejű előrejelző végpont
- MLflow integráció modell betöltéshez
- Bemeneti validáció és hibakezelés

### Streamlit Irányítópult
- Felhasználói felület előrejelzésekhez
- Interaktív adatvizualizáció
- Modellmonitorozási jelentések
- Igény szerinti riportgenerálás

### MLflow
- Modellkövetés és verziózás
- Kísérletek összehasonlítása
- Modellregisztráció
- Központosított artefakt tárolás

### Evidently AI
- Modellmonitorozási riportok
- Adateltolódás észlelése
- Teljesítménykövetés
- Automatizált riportgenerálás

### Airflow
- Munkafolyamat automatizálás
- Ütemezett újratanítás
- DAG-alapú pipeline kezelés
- Hibakezelés és értesítések

## Architektúra

![Iris MLOps Architektúra](./architecture.png)

### Adatáramlás
1. A tanító jegyzetfüzetek létrehozzák a modelleket, és regisztrálják azokat az MLflow-ban
2. A modellek és artefaktok verziózva és tárolva lesznek
3. A FastAPI betölti a modellt előrejelzéshez
4. A Streamlit irányítópult kommunikál az API-val és az MLflow-val
5. Az Airflow rendszeresen ütemezi az újratanítást és monitorozást
6. A felhasználók az API-t és az irányítópultot is használhatják

## Indítás

### Előfeltételek

- Docker és Docker Compose
- Python 3.9+
- Git

### A projekt futtatása

1. Klónozd a repót:

```bash
git clone <repository-url>
cd iris-mlops-project
```

2. Indítsd el a Docker Compose fájlt:

```bash
docker-compose up -d
```

3. Szolgáltatások elérhetősége:

- Streamlit Dashboard: http://localhost:8501
- FastAPI API: http://localhost:8000
- MLflow szerver: http://localhost:5000
- Airflow felület: http://localhost:8080

### Kezdeti beállítások
A szolgáltatások elindítása után:

1. Generáld le az elsődleges modellt (ha még nincs):

```bash
python generate_initial_model.py
```

2. Inicializáld az MLflow kísérletet (ha szükséges):

```bash
python init_mlflow.py
```
3. Generálj modellmonitorozási riportot:

```bash
python generate_report.py
```

## Szolgáltatások használata
### FastAPI
- Előrejelzések: http://localhost:8000/predict

- API dokumentáció: http://localhost:8000/docs

### Streamlit Irányítópult
- Interaktív UI: http://localhost:8501

- Navigáció az oldalsáv segítségével

### MLflow
- Kísérletek és modellek: http://localhost:5000

- Futtatások összehasonlítása, artefaktok letöltése

### Airflow
- Felület: http://localhost:8080

- Alapértelmezett belépési adatok: airflow/airflow

- Aktiváld a szükséges DAG-okat

### Tesztek futtatása

```bash
pytest tests/
```

### Projektstruktúra
```bash
iris-mlops-beadando
├── app/                    # FastAPI alkalmazás
│   └── main.py             # API megvalósítás
├── airflow/                # Airflow konfiguráció
│   ├── dags/               # Airflow DAG-ok
│   │   └── iris_dag.py     # Iris pipeline DAG
│   ├── config/             # Airflow konfigurációs fájlok
│   ├── logs/               # Airflow naplók
│   └── plugins/            # Airflow pluginek
├── mlruns/                 # MLflow nyomkövetési adatok
├── mlartifacts/            # MLflow artefaktok
├── notebooks/              # Jupyter jegyzetfüzetek
│   └── 01_training.ipynb   # Modell tanító jegyzetfüzet
├── streamlit_app/          # Streamlit irányítópult
│   └── dashboard.py        # Irányítópult megvalósítás
├── tests/                  # Tesztek
│   └── test_api.py         # API tesztek
├── .env                    # Környezeti változók
├── docker-compose.yml      # Docker Compose konfiguráció
├── Dockerfile              # Dockerfile FastAPI-hoz
├── mlflow.Dockerfile       # Dockerfile MLflow-hoz
├── streamlit.Dockerfile    # Dockerfile Streamlithez
├── create_experiment.py    # MLflow kísérlet létrehozó script
├── generate_initial_model.py # Kezdeti modell generálása
├── generate_report.py      # Evidently riport generálása
├── init_mlflow.py          # MLflow inicializálás
├── model.joblib            # Elmentett modell
├── README.md               # Ez a fájl
├── requirements.txt        # Python csomagok
├── scaler.joblib           # Elmentett skálázó
└── system_info.json        # Rendszerinformáció
```

## Modell Információk
### Adathalmaz
A modell a klasszikus Iris adathalmazt használja, amely három íriszvirág fajra vonatkozó méréseket tartalmaz:

- Setosa

- Versicolor

- Virginica

Jellemzők: csészelevél hossza/szélessége, szirom hossza/szélessége.

### Modell Architektúra
A rendszer egy scikit-learn osztályozót (általában Random Forest) használ, amely az Iris adathalmazon van betanítva. A jellemzők skálázása történik az előrejelzések előtt.

### Teljesítménymutatók
A modellt a következő metrikákkal értékeljük:

- Pontosság (Accuracy)

- Precizitás (Precision)

- Visszahívás (Recall)

- F1 Pontszám