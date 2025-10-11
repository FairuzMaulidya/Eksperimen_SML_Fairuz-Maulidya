# Import Library
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import os

# Melakukan proses automate preprocessing
def automate_preprocess(dataset, target, save_path, file_path):
    # Membuat kolom status berdasarkan rata-rata nilai
    dataset['status'] = dataset[['math score', 'reading score', 'writing score']].mean(axis=1)
    dataset['status'] = dataset['status'].apply(lambda x: 'Failed' if x <= 60 else 'Pass')

    column = dataset.columns.drop(target)
    df_head = pd.DataFrame(columns=column)
    df_head.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    student_cat = dataset.select_dtypes(include='object').drop(columns=[target], errors='ignore').columns
    student_num = dataset.select_dtypes(include='number').columns

    num_transform = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    cat_transform = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transform, student_num),
            ('cat', cat_transform, student_cat)
        ]
    )

    X = dataset.drop(columns=[target])
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test

# Menjalankan Fungsinya
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "StudentsPerformance_raw.csv")
    save_path = os.path.join(base_dir, "StudentsPerformance_preprocessing.joblib")
    file_path = os.path.join(base_dir, "Column-StudentsPerformance_preprocessing.csv")

    student = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = automate_preprocess(
        dataset=student,
        target='status',
        save_path=save_path,
        file_path=file_path
    )
