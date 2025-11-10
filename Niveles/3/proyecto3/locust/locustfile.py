from locust import HttpUser, task, between
import random

class DiabetesAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_readmission(self):
        races = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other/Unknown"]
        genders = ["Male", "Female"]
        ages = ["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)"]
        glu_levels = ["None", "Norm", ">200", ">300"]
        a1c_levels = ["None", "Norm", ">7", ">8"]
        
        payload = {
            "race": random.choice(races),
            "gender": random.choice(genders),
            "age": random.choice(ages),
            "admission_type_id": random.randint(1, 8),
            "discharge_disposition_id": random.randint(1, 10),
            "admission_source_id": random.randint(1, 9),
            "time_in_hospital": random.randint(1, 10),
            "payer_code": "MC",
            "medical_specialty": "Cardiology",
            "num_lab_procedures": random.randint(20, 80),
            "num_procedures": random.randint(0, 6),
            "num_medications": random.randint(5, 30),
            "number_outpatient": random.randint(0, 5),
            "number_emergency": random.randint(0, 3),
            "number_inpatient": random.randint(0, 5),
            "diag_1": "250.83",
            "diag_2": "401.9",
            "diag_3": "428.0",
            "number_diagnoses": random.randint(5, 12),
            "max_glu_serum": random.choice(glu_levels),
            "A1Cresult": random.choice(a1c_levels),
            "change": random.choice(["no", "yes"]),
            "diabetesMed": random.choice(["no", "yes"])
        }
        
        self.client.post("/predict", json=payload)
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
    
    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")
