#Mengimport Library
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import os

def automate_preprocess(dataset, save_path, file_path): 
    #Melakukan bining untuk menentukan target
    student['status'] = student[['math score','reading score','writing score']].mean(axis=1).apply(lambda x: 'Failed' if x <= 60 else 'Pass')

    #Menentukan kolom kategorik
    student_cat = student.select_dtypes(include='object')

    #Melakukan encode pada data kategorikal
    encode = LabelEncoder()
    for col in student_cat.columns:
        student[col] = encode.fit_transform(student[col])

    #Melakukan scale pada data 
    score = ['math score','reading score','writing score']
    scale = MinMaxScaler()
    student[score] = scale.fit_transform(student[score])
    
    #Melakukan split data
    X = student.drop('status', axis=1)
    y = student['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Menyimpan kolom header tanpa data dan kolom target
    target = ['status']
    column = dataset.columns.drop(target)
    pd.DataFrame(columns=column).to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    #Menyimpan hasil preprocessing
    file_preprocess = {
        "scaler": scale,
        "label_encoders": encode,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    dump(file_preprocess, save_path)
    print(f"Preprocessing dan data split disimpan ke: {save_path}")

    return file_preprocess

#Mententukan nama file
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "..", "StudentsPerformance_raw.csv")
save_path = os.path.join(base_dir, "StudentsPerformance_preprocessing.joblib")
file_path = os.path.join(base_dir, "Column-StudentsPerformance_preprocessing.csv")

#Menjalankan fungsi automate preprocessing
student = pd.read_csv(dataset_path)
data = automate_preprocess(
        dataset=student,
        save_path=save_path,
        file_path=file_path
    )
